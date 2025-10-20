from .tensorrt_convert import DYNAMIC_TRT_MODEL_CONVERSION
from .tensorrt_convert import STATIC_TRT_MODEL_CONVERSION
from .tensorrt_loader import TrTUnet
from .tensorrt_loader import TensorRTLoader

# CRITICAL: Apply Lumina2 unpatchify patch for TensorRT compatibility
# This fixes vertical line artifacts caused by .view() vs .reshape() issues
try:
    from .lumina2_vae_patch import patch_lumina2_unpatchify
    if patch_lumina2_unpatchify():
        print("[ComfyUI_TensorRT] ✅ Lumina2 unpatchify patch applied successfully")
    else:
        print("[ComfyUI_TensorRT] ⚠️  Lumina2 unpatchify patch not applied (model not loaded yet)")
except Exception as e:
    print(f"[ComfyUI_TensorRT] ⚠️  Lumina2 unpatchify patch failed: {e}")

# Enable VAE debugging for artifact analysis
try:
    from .vae_debug_hook import enable_vae_debug
    enable_vae_debug()
    print("[ComfyUI_TensorRT] ✅ VAE Debug hook enabled - spatial pattern analysis active")
except Exception as e:
    print(f"[ComfyUI_TensorRT] ⚠️  VAE Debug hook failed to load: {e}")

NODE_CLASS_MAPPINGS = { "DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION, "STATIC_TRT_MODEL_CONVERSION": STATIC_TRT_MODEL_CONVERSION, "TensorRTLoader": TensorRTLoader }


NODE_DISPLAY_NAME_MAPPINGS = { "DYNAMIC_TRT_MODEL_CONVERSION": "DYNAMIC TRT_MODEL CONVERSION", "STATIC TRT_MODEL CONVERSION": STATIC_TRT_MODEL_CONVERSION, "TensorRTLoader": "TensorRT Loader" }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']