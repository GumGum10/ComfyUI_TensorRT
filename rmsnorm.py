import torch
import comfy.model_management
import numbers
import logging

# Register ONNX symbolic for RMSNorm compatible with PyTorch 2.8
# aten::rms_norm signature: (input, normalized_shape, weight, eps)
@torch.onnx.symbolic_helper.parse_args("v", "is", "v", "f")
def symbolic_rms_norm(g, input, normalized_shape, weight, eps):
    # Determine reduction dims from normalized_shape length (last K dims)
    try:
        k = len(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else int(normalized_shape)
    except Exception:
        # Fallback to last dim
        k = 1
    dims = list(range(-k, 0)) if k > 0 else [-1]

    # x^2 -> mean over dims -> +eps -> rsqrt
    squared = g.op("Mul", input, input)
    mean_squares = g.op("ReduceMean", squared, axes_i=dims, keepdims_i=1)

    # Use float32 eps constant and cast to input dtype to avoid unsupported bf16 constants
    eps_value = 1e-6 if (eps is None) else float(eps)
    eps_const = g.op("Constant", value_t=torch.tensor(eps_value, dtype=torch.float32))
    eps_like = g.op("CastLike", eps_const, mean_squares)
    add_eps = g.op("Add", mean_squares, eps_like)
    rsqrt = g.op("Reciprocal", g.op("Sqrt", add_eps))

    # Normalize input
    normalized = g.op("Mul", input, rsqrt)

    # Optionally apply weight (broadcast on trailing dims)
    try:
        is_none = torch.onnx.symbolic_helper._is_none  # type: ignore[attr-defined]
    except Exception:
        def is_none(x):
            return x is None

    if (weight is not None) and (not is_none(weight)):
        normalized = g.op("Mul", normalized, weight)

    return normalized

# Register the symbolic function for opset 17+
if hasattr(torch.ops, "aten") and hasattr(torch.ops.aten, "rms_norm"):
    try:
        torch.onnx.register_custom_op_symbolic("aten::rms_norm", symbolic_rms_norm, 17)
        # Also register for a few newer opsets for safety
        for opset in (18, 19):
            try:
                torch.onnx.register_custom_op_symbolic("aten::rms_norm", symbolic_rms_norm, opset)
            except Exception:
                pass
    except Exception as e:
        logging.debug(f"[RMSNorm] Failed to register ONNX symbolic: {e}")

RMSNorm = None

try:
    rms_norm_torch = torch.nn.functional.rms_norm
    RMSNorm = torch.nn.RMSNorm
except:
    rms_norm_torch = None
    logging.warning("Please update pytorch to use native RMSNorm")


def rms_norm(x, weight=None, eps=1e-6):
    if rms_norm_torch is not None and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # Log input state
        logging.info(f"[RMSNorm] Input tensor shape: {x.shape}, dtype: {x.dtype}")
        if weight is not None:
            logging.info(f"[RMSNorm] Weight tensor shape: {weight.shape}, dtype: {weight.dtype}")
        
        if weight is None:
            logging.info("[RMSNorm] No weight provided, using native RMSNorm")
            return rms_norm_torch(x, (x.shape[-1],), eps=eps)
        else:
            # Handle dimension mismatch for Lumina2 models
            input_dim = x.shape[-1]
            if weight.shape[-1] != input_dim:
                logging.info(f"[RMSNorm] Dimension mismatch detected:")
                logging.info(f"[RMSNorm] - Input dim: {input_dim}")
                logging.info(f"[RMSNorm] - Weight dim: {weight.shape[-1]}")
                if weight.shape[-1] > input_dim:
                    logging.info(f"[RMSNorm] Weight larger than input, truncating from {weight.shape[-1]} to {input_dim}")
                    # Extract primary submatrix for the linear transform
                    primary_weight = weight[..., :input_dim]
                    logging.info(f"[RMSNorm] After first truncation: {primary_weight.shape}")
                    
                    if len(weight.shape) > 1 and weight.shape[0] > input_dim:
                        logging.info(f"[RMSNorm] 2D weight larger than input, truncating from {weight.shape[0]} to {input_dim}")
                        primary_weight = primary_weight[:input_dim, :]
                        logging.info(f"[RMSNorm] After 2D truncation: {primary_weight.shape}")
                else:
                    # Pad weight to match input dimension
                    pad_size = input_dim - weight.shape[-1]
                    logging.info(f"[RMSNorm] Weight smaller than input, padding from {weight.shape[-1]} to {input_dim} (pad_size: {pad_size})")
                    primary_weight = torch.nn.functional.pad(weight, (0, pad_size), mode='constant', value=0)
                    logging.info(f"[RMSNorm] After first padding: {primary_weight.shape}")
                    
                    if len(weight.shape) > 1:
                        logging.info(f"[RMSNorm] Adding 2D padding for shape {primary_weight.shape}")
                        primary_weight = torch.nn.functional.pad(primary_weight, (0, 0, 0, pad_size), mode='constant', value=0)
                        logging.info(f"[RMSNorm] After 2D padding: {primary_weight.shape}")
                
                weight = primary_weight
                logging.info(f"[RMSNorm] Final weight shape: {weight.shape}")

            # Ensure weight has correct shape for normalization
            if len(weight.shape) == 1:
                weight = weight.view(1, 1, -1)
            elif len(weight.shape) == 2:
                weight = weight.view(1, *weight.shape)

            # Apply normalization
            return rms_norm_torch(x, weight.shape, weight=comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)
    else:
        r = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        if weight is None:
            return r
        else:
            # Handle dimension mismatch for Lumina2 models
            input_dim = x.shape[-1]
            if weight.shape[-1] != input_dim:
                logging.info(f"RMSNorm dimension mismatch - weight: {weight.shape}, input: {x.shape}")
                if weight.shape[-1] > input_dim:
                    logging.info(f"[RMSNorm] Weight larger than input, truncating from {weight.shape[-1]} to {input_dim}")
                    # Extract primary submatrix for the linear transform
                    primary_weight = weight[..., :input_dim]
                    logging.info(f"[RMSNorm] After first truncation: {primary_weight.shape}")
                    
                    if len(weight.shape) > 1 and weight.shape[0] > input_dim:
                        logging.info(f"[RMSNorm] 2D weight larger than input, truncating from {weight.shape[0]} to {input_dim}")
                        primary_weight = primary_weight[:input_dim, :]
                        logging.info(f"[RMSNorm] After 2D truncation: {primary_weight.shape}")
                else:
                    # Pad weight to match input dimension
                    pad_size = input_dim - weight.shape[-1]
                    logging.info(f"[RMSNorm] Weight smaller than input, padding from {weight.shape[-1]} to {input_dim} (pad_size: {pad_size})")
                    primary_weight = torch.nn.functional.pad(weight, (0, pad_size), mode='constant', value=0)
                    logging.info(f"[RMSNorm] After first padding: {primary_weight.shape}")
                    
                    if len(weight.shape) > 1:
                        logging.info(f"[RMSNorm] Adding 2D padding for shape {primary_weight.shape}")
                        primary_weight = torch.nn.functional.pad(primary_weight, (0, 0, 0, pad_size), mode='constant', value=0)
                        logging.info(f"[RMSNorm] After 2D padding: {primary_weight.shape}")
                
                weight = primary_weight
                logging.info(f"[RMSNorm] Final weight shape: {weight.shape}")

            # Ensure weight has correct shape for normalization
            if len(weight.shape) == 1:
                weight = weight.view(1, 1, -1)
            elif len(weight.shape) == 2:
                weight = weight.view(1, *weight.shape)

            return r * comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)


if RMSNorm is None:
    class RMSNorm(torch.nn.Module):
        def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                # mypy error: incompatible types in assignment
                normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = torch.nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("weight", None)
            self.bias = None

        def forward(self, x):
            return rms_norm(x, self.weight, self.eps)
