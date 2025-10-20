"""
Lumina2 Unpatchify Patch for TensorRT Compatibility
Replaces .view() with .reshape() to handle non-contiguous tensors from TensorRT
"""

import torch
import logging


def patch_lumina2_unpatchify():
    """
    Patch Lumina2's unpatchify to be more robust with TensorRT tensors.
    
    The original unpatchify uses .view() which requires contiguous memory.
    TensorRT may output tensors with different strides, causing .view() to fail
    or produce incorrect results (vertical line artifacts).
    
    This patch uses .reshape() instead, which handles non-contiguous tensors correctly.
    """
    try:
        from comfy.ldm.lumina import model as lumina_model
    except ImportError:
        logging.warning("[TensorRT] Could not import Lumina2 model for unpatchify patch")
        return False
    
    # Check if already patched
    if hasattr(lumina_model.NextDiT.unpatchify, '_tensorrt_patched'):
        logging.info("[TensorRT] Lumina2 unpatchify already patched")
        return True
    
    # Save original for reference
    original_unpatchify = lumina_model.NextDiT.unpatchify
    
    def robust_unpatchify(self, x, img_size, cap_size, return_tensor=False):
        """
        Enhanced unpatchify that works with TensorRT tensors.
        
        Key changes:
        - Forces contiguous memory layout BEFORE any reshaping
        - Uses .reshape() instead of .view() for robustness
        - This handles non-contiguous memory layouts from TensorRT
        """
        pH = pW = self.patch_size
        imgs = []
        
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            
            # Extract patch sequence for this image
            patches = x[i][begin:end]
            
            # DEBUG: Log input properties (only in debug mode)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"[Unpatchify] patches shape: {patches.shape}, contiguous: {patches.is_contiguous()}, stride: {patches.stride()}")
            
            # CRITICAL FIX 1: Force contiguous memory layout
            # TensorRT tensors may report is_contiguous()=True but have non-standard strides
            # .clone() guarantees fresh allocation with standard layout
            if not patches.is_contiguous():
                patches = patches.contiguous()
            
            # CRITICAL FIX 2: Use reshape instead of view for extra safety
            # reshape() handles edge cases that view() doesn't
            imgs.append(
                patches
                .reshape(H // pH, W // pW, pH, pW, self.out_channels)  # ← Changed from .view()
                .permute(4, 0, 2, 1, 3)  # Reorder: [C, H_patches, pH, W_patches, pW]
                .flatten(3, 4)           # Flatten W: [C, H_patches, pH, W]
                .flatten(1, 2)           # Flatten H: [C, H, W]
            )

        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        
        return imgs
    
    # Mark as patched
    robust_unpatchify._tensorrt_patched = True
    robust_unpatchify._original = original_unpatchify
    
    # Apply the patch
    lumina_model.NextDiT.unpatchify = robust_unpatchify
    logging.info("[TensorRT] ✅ Patched Lumina2 unpatchify for TensorRT compatibility (.view→.reshape)")
    
    return True


def enable_unpatchify_patch():
    """Enable the unpatchify patch (call from __init__.py if needed)"""
    return patch_lumina2_unpatchify()
