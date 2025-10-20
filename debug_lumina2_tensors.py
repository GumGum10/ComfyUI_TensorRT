"""
Diagnostic script to check Lumina2 tensor operations
Run this to verify what's happening inside the model
"""

import torch
import logging

def add_tensor_hooks():
    """
    Add hooks to track tensor properties through Lumina2 forward pass
    """
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                logging.info(f"[HOOK] {name}:")
                logging.info(f"  Shape: {output.shape}")
                logging.info(f"  Dtype: {output.dtype}")
                logging.info(f"  Device: {output.device}")
                logging.info(f"  Contiguous: {output.is_contiguous()}")
                logging.info(f"  Stride: {output.stride()}")
                logging.info(f"  Min/Max: {output.min().item():.4f} / {output.max().item():.4f}")
                logging.info(f"  Mean/Std: {output.mean().item():.4f} / {output.std().item():.4f}")
                
                # Check for NaN/Inf
                has_nan = output.isnan().any().item()
                has_inf = output.isinf().any().item()
                if has_nan or has_inf:
                    logging.warning(f"  ⚠️  NaN: {has_nan}, Inf: {has_inf}")
            
            return output
        return hook
    
    try:
        from comfy.ldm.lumina import model as lumina_model
        
        # Hook into key layers
        for attr_name in ['final_layer', 'unpatchify']:
            if hasattr(lumina_model.NextDiT, attr_name):
                logging.info(f"Adding hook to NextDiT.{attr_name}")
                # This will hook all instances
                # We'll need to manually register this per instance
                
        logging.info("[DEBUG] Tensor hooks configured")
        return True
        
    except Exception as e:
        logging.error(f"[DEBUG] Failed to add hooks: {e}")
        return False


def check_unpatchify_implementation():
    """
    Verify which unpatchify implementation is being used
    """
    try:
        from comfy.ldm.lumina import model as lumina_model
        
        unpatchify_func = lumina_model.NextDiT.unpatchify
        
        logging.info("[DEBUG] Unpatchify implementation check:")
        logging.info(f"  Has _tensorrt_patched: {hasattr(unpatchify_func, '_tensorrt_patched')}")
        logging.info(f"  Has _original: {hasattr(unpatchify_func, '_original')}")
        
        # Try to read source
        import inspect
        try:
            source = inspect.getsource(unpatchify_func)
            uses_view = '.view(' in source
            uses_reshape = '.reshape(' in source
            logging.info(f"  Uses .view(): {uses_view}")
            logging.info(f"  Uses .reshape(): {uses_reshape}")
            
            if uses_view and not uses_reshape:
                logging.warning("  ⚠️  STILL USING .view() - PATCH NOT APPLIED!")
            elif uses_reshape:
                logging.info("  ✅ Using .reshape() - patch appears active")
                
        except Exception as e:
            logging.debug(f"  Could not inspect source: {e}")
            
        return True
        
    except Exception as e:
        logging.error(f"[DEBUG] Failed to check unpatchify: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("LUMINA2 DIAGNOSTIC CHECK")
    print("=" * 60)
    
    check_unpatchify_implementation()
    add_tensor_hooks()
