"""
ONNX Export Patch for Lumina2 NextDiT Model
This patches ComfyUI's Lumina2 implementation to be ONNX-traceable
"""

import torch
import torch.nn as nn
import logging


def patch_lumina2_for_onnx_export():
    """
    Monkey-patch ComfyUI's Lumina2 model to make it ONNX-exportable.
    
    The main issues are:
    1. torch.arange() calls with tensor arguments (not supported in ONNX tracing)
    2. Dynamic list comprehensions over tensors
    3. Variable-length sequence handling
    4. transformer_options parameter during tracing
    """
    import comfy.ldm.lumina.model as lumina_model
    
    # Save original method
    original_patchify_and_embed = lumina_model.NextDiT.patchify_and_embed
    
    def onnx_compatible_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, transformer_options={}):
        """ONNX-compatible version of patchify_and_embed"""
        
        # Handle list vs tensor input
        if isinstance(x, torch.Tensor):
            # Single tensor input - convert to list
            bsz = x.shape[0]
            x = [x[i] for i in range(bsz)]
        else:
            bsz = len(x)
        
        pH = pW = self.patch_size
        device = x[0].device
        dtype = x[0].dtype
        
        # FIXED: Convert tensor to int for ONNX compatibility
        if cap_mask is not None:
            # Use .item() to convert scalar tensor to Python int
            if isinstance(cap_mask, torch.Tensor):
                l_effective_cap_len = [int(cap_mask[i].sum().item()) for i in range(bsz)]
            else:
                l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        else:
            # FIXED: Ensure num_tokens is Python int
            if isinstance(num_tokens, torch.Tensor):
                num_tokens_val = int(num_tokens.item()) if num_tokens.numel() == 1 else int(num_tokens[0].item())
            else:
                num_tokens_val = int(num_tokens)
            l_effective_cap_len = [num_tokens_val] * bsz
        
        if cap_mask is not None and not torch.is_floating_point(cap_mask):
            cap_mask = (cap_mask - 1).to(dtype) * torch.finfo(dtype).max
        
        img_sizes = [(int(img.size(1)), int(img.size(2))) for img in x]
        l_effective_img_len = [(H // pH) * (W // pW) for (H, W) in img_sizes]
        
        max_seq_len = max(
            (cap_len + img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len))
        )
        max_cap_len = max(l_effective_cap_len)
        max_img_len = max(l_effective_img_len)
        
        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)
        
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            H, W = img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW
            
            # FIXED: Use int() to convert Python int explicitly for ONNX
            # This avoids the tensor argument issue
            position_ids[i, :cap_len, 0] = torch.arange(0, cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:cap_len+img_len, 0] = cap_len
            
            # FIXED: Precompute row/col indices as Python ints
            row_ids = torch.arange(0, H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(0, W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            position_ids[i, cap_len:cap_len+img_len, 1] = row_ids
            position_ids[i, cap_len:cap_len+img_len, 2] = col_ids
        
        freqs_cis = self.rope_embedder(position_ids).movedim(1, 2).to(dtype)
        
        # Rest of the function continues as original...
        cap_freqs_cis_shape = list(freqs_cis.shape)
        cap_freqs_cis_shape[1] = cap_feats.shape[1]
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)
        
        img_freqs_cis_shape = list(freqs_cis.shape)
        img_freqs_cis_shape[1] = max_img_len
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)
        
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len:cap_len+img_len]
        
        # FIXED: Refine context - DON'T pass transformer_options during ONNX tracing
        # Check if we're in tracing mode
        is_tracing = torch.jit.is_tracing() or torch.jit.is_scripting()
        
        for layer in self.context_refiner:
            if is_tracing:
                # During ONNX export, call without transformer_options
                cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)
            else:
                # Normal runtime, pass transformer_options
                cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis, transformer_options=transformer_options)
        
        # Refine image
        flat_x = []
        for i in range(bsz):
            img = x[i]
            C, H, W = img.size()
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 2, 4, 0).flatten(2).flatten(0, 1)
            flat_x.append(img)
        x = flat_x
        
        padded_img_embed = torch.zeros(bsz, max_img_len, x[0].shape[-1], device=device, dtype=x[0].dtype)
        padded_img_mask = torch.zeros(bsz, max_img_len, dtype=dtype, device=device)
        for i in range(bsz):
            padded_img_embed[i, :l_effective_img_len[i]] = x[i]
            padded_img_mask[i, l_effective_img_len[i]:] = -torch.finfo(dtype).max
        
        padded_img_embed = self.x_embedder(padded_img_embed)
        padded_img_mask = padded_img_mask.unsqueeze(1)
        
        # FIXED: Same for noise_refiner - don't pass transformer_options during tracing
        for layer in self.noise_refiner:
            if is_tracing:
                # During ONNX export, call without transformer_options
                padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t)
            else:
                # Normal runtime
                padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t, transformer_options=transformer_options)
        
        if cap_mask is not None:
            mask = torch.zeros(bsz, max_seq_len, dtype=dtype, device=device)
            mask[:, :max_cap_len] = cap_mask[:, :max_cap_len]
        else:
            mask = None
        
        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=x[0].dtype)
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            padded_full_embed[i, cap_len:cap_len+img_len] = padded_img_embed[i, :img_len]
        
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis
    
    # Apply the patch
    lumina_model.NextDiT.patchify_and_embed = onnx_compatible_patchify_and_embed
    logging.info("[TensorRT] ✅ Applied ONNX compatibility patch to Lumina2 NextDiT")
    
    return True