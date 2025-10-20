#Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)

import torch
import os
import logging

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths

# Import VAE debug hook
try:
    from .vae_debug_hook import get_vae_debug_hook
    VAE_DEBUG_AVAILABLE = True
except ImportError:
    VAE_DEBUG_AVAILABLE = False
    logging.warning("[TensorRT] VAE debug hook not available")

if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.models_dir, "tensorrt"))
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.models_dir, "tensorrt")], {".engine"})

import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

# Is there a function that already exists for this?
def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16

class TrTUnet:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.dtype = torch.float16

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, x, timesteps, context, y=None, control=None, transformer_options=None, **kwargs):
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        print(f"\n{'='*60}")
        print(f"[TensorRT Loader Debug]")
        print(f"Input context shape: {context.shape}")
        print(f"Expected by engine:")

        # Add these lines:
        try:
            prof_min, prof_opt, prof_max = self.engine.get_tensor_profile_shape("context", 0)
            print(f"  Min: {prof_min}")
            print(f"  Opt: {prof_opt}")
            print(f"  Max: {prof_max}")
        except Exception as e:
            print(f"  Error: {e}")

        print(f"{'='*60}\n")
        # DEBUG: Print what we're actually receiving
        print(f"\n{'='*60}")
        print(f"[TensorRT Loader Debug]")
        print(f"Input context shape: {context.shape}")  # What we receive
        print(f"Expected by engine:")


        # FIXED: Proper context handling without incorrect padding
        # For Lumina2, context should already be the correct dimension (2304) from text encoder
        # No need to pad or truncate - just validate
        try:
            ctx_name = "context"
            if ctx_name in model_inputs:
                # Get engine's expected shape from profile
                prof_min, prof_opt, prof_max = self.engine.get_tensor_profile_shape(ctx_name, 0)
                
                ctx = model_inputs[ctx_name]
                B, L, C = ctx.shape
                
                # Validate feature dimension matches
                expected_feat = prof_min[2] if len(prof_min) >= 3 else None
                if expected_feat is not None and expected_feat != -1 and C != expected_feat:
                    raise ValueError(f"Context feature dimension mismatch: got {C}, expected {expected_feat}")
                
                # Handle sequence length by padding/truncating if needed
                min_len = prof_min[1] if len(prof_min) >= 2 else None
                max_len = prof_max[1] if len(prof_max) >= 2 else None
                
                if (min_len is not None and min_len != -1) or (max_len is not None and max_len != -1):
                    # Truncate if longer than max
                    if max_len is not None and max_len != -1 and L > max_len:
                        ctx = ctx[:, :max_len, :]
                        L = max_len
                    # Pad if shorter than min
                    if min_len is not None and min_len != -1 and L < min_len:
                        pad_L = min_len - L
                        # Use zero padding for simplicity
                        pad_tensor = torch.zeros(B, pad_L, C, dtype=ctx.dtype, device=ctx.device)
                        ctx = torch.cat([ctx, pad_tensor], dim=1)

                model_inputs[ctx_name] = ctx.contiguous()
        except Exception as e:
            # Best-effort; if inspection fails, continue and let TRT report a clear error
            print(f"Warning: Context shape adjustment failed: {e}")
            pass

        if y is not None:
            model_inputs["y"] = y

        # # FIXED: Handle num_tokens for Lumina2
        # num_tokens = kwargs.get("num_tokens", None)
        # if num_tokens is not None:
        #     # num_tokens should be a scalar or 1D tensor
        #     if isinstance(num_tokens, int):
        #         num_tokens = torch.tensor([num_tokens], dtype=torch.int32, device=x.device)
        #     elif isinstance(num_tokens, torch.Tensor):
        #         if num_tokens.dim() == 0:
        #             num_tokens = num_tokens.unsqueeze(0)
        #         num_tokens = num_tokens.to(dtype=torch.int32, device=x.device)
        #     model_inputs["num_tokens"] = num_tokens

        for i in range(len(model_inputs), self.engine.num_io_tensors - 1):
            name = self.engine.get_tensor_name(i)
            if name in kwargs:
                model_inputs[name] = kwargs[name]

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]

        #Split batch if our batch is bigger than the max batch size the trt engine supports
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        #for dynamic profile case where the dynamic params are -1
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape, 
                          device=x.device, 
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                x_input = model_inputs_converted[k]
                self.context.set_tensor_address(k, x_input[(x_input.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        # stream.synchronize() #don't need to sync stream since it's the default torch one
        
        # ============================================================================
        # CRITICAL FIX FOR VERTICAL LINE ARTIFACTS
        # ============================================================================
        # TensorRT may output tensors with non-standard memory layouts (strides)
        # that cause Lumina2's unpatchify .view() operation to produce artifacts.
        # 
        # Root cause: .view() requires contiguous memory with standard strides
        # Solution: Force proper memory layout using .clone() which guarantees
        #           fresh allocation with standard C-contiguous layout
        # ============================================================================
        
        # DEBUG: Log TensorRT raw output properties
        print(f"\n{'='*60}")
        print(f"[TensorRT VAE Debug] RAW OUTPUT from TensorRT:")
        print(f"  Shape: {out.shape}")
        print(f"  Dtype: {out.dtype}")
        print(f"  Device: {out.device}")
        print(f"  Is contiguous: {out.is_contiguous()}")
        print(f"  Stride: {out.stride()}")
        print(f"  Min value: {out.min().item():.6f}")
        print(f"  Max value: {out.max().item():.6f}")
        print(f"  Mean value: {out.mean().item():.6f}")
        print(f"  Std value: {out.std().item():.6f}")
        print(f"  Has NaN: {torch.isnan(out).any().item()}")
        print(f"  Has Inf: {torch.isinf(out).any().item()}")
        
        # CRITICAL FIX: Force proper memory layout with .clone()
        # This creates a new tensor with guaranteed standard C-contiguous layout
        # .contiguous() alone may not change stride if tensor appears contiguous
        # but has non-standard stride pattern from TensorRT
        original_dtype = out.dtype
        out_fixed = out.to(dtype=torch.float32).clone()
        
        # Verify the fix worked
        expected_stride = (
            out_fixed.shape[1] * out_fixed.shape[2] * out_fixed.shape[3],  # Batch stride
            out_fixed.shape[2] * out_fixed.shape[3],                        # Channel stride
            out_fixed.shape[3],                                              # Height stride
            1                                                                # Width stride
        )
        
        print(f"\n[TensorRT VAE Debug] FIXED OUTPUT (after clone + fp32):")
        print(f"  Shape: {out_fixed.shape}")
        print(f"  Dtype: {out_fixed.dtype}")
        print(f"  Device: {out_fixed.device}")
        print(f"  Is contiguous: {out_fixed.is_contiguous()}")
        print(f"  Stride: {out_fixed.stride()}")
        print(f"  Expected stride: {expected_stride}")
        print(f"  Stride OK: {out_fixed.stride() == expected_stride}")
        print(f"  Memory format: {out_fixed.stride()}")
        print(f"{'='*60}\n")
        
        # Advanced VAE debugging if available
        if VAE_DEBUG_AVAILABLE:
            debug_hook = get_vae_debug_hook()
            if debug_hook is not None:
                debug_hook.analyze_latent(out_fixed, source="TensorRT")
        
        return out_fixed

    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}


class TensorRTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("tensorrt"), ),
                             "model_type": (["sdxl_base", "sdxl_refiner", "sd1.x", "sd2.x-768v", "svd", "sd3", "auraflow", "flux_dev", "flux_schnell", "lumina2"], ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")
        unet = TrTUnet(unet_path)
        if model_type == "sdxl_base":
            conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXL(conf)
        elif model_type == "sdxl_refiner":
            conf = comfy.supported_models.SDXLRefiner(
                {"adm_in_channels": 2560})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXLRefiner(conf)
        elif model_type == "sd1.x":
            conf = comfy.supported_models.SD15({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf)
        elif model_type == "sd2.x-768v":
            conf = comfy.supported_models.SD20({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf, model_type=comfy.model_base.ModelType.V_PREDICTION)
        elif model_type == "svd":
            conf = comfy.supported_models.SVD_img2vid({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "sd3":
            conf = comfy.supported_models.SD3({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "auraflow":
            conf = comfy.supported_models.AuraFlow({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "flux_dev":
            conf = comfy.supported_models.Flux({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
            unet.dtype = torch.bfloat16 #TODO: autodetect
        elif model_type == "flux_schnell":
            conf = comfy.supported_models.FluxSchnell({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
            unet.dtype = torch.bfloat16 #TODO: autodetect
        elif model_type == "lumina2":
            # FIXED: Proper Lumina2 configuration
            from comfy.supported_models import Lumina2 as Lumina2Config
            conf = Lumina2Config({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
            unet.dtype = torch.bfloat16  # Match Lumina2 default
        model.diffusion_model = unet
        model.memory_required = lambda *args, **kwargs: 0 #always pass inputs batched up as much as possible, our TRT code will handle batch splitting

        return (comfy.model_patcher.ModelPatcher(model,
                                                 load_device=comfy.model_management.get_torch_device(),
                                                 offload_device=comfy.model_management.unet_offload_device()),)

NODE_CLASS_MAPPINGS = {
    "TensorRTLoader": TensorRTLoader,
}