"""
ONNX Export Patch for Lumina2 NextDiT Model - FINAL FIX
Based on actual ComfyUI source code analysis
"""

import torch
import torch.nn as nn
import logging


def patch_lumina2_for_onnx_export():
    """
    Patch ComfyUI's Lumina2 to be ONNX-exportable.
    Based on actual source: comfy/ldm/lumina/model.py
    """
    import comfy.ldm.lumina.model as lumina_model
    
    original_patchify_and_embed = lumina_model.NextDiT.patchify_and_embed
    
    def onnx_compatible_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens):
        """
        ONNX-compatible version matching the original logic from line 421-497
        """
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device
        dtype = x[0].dtype

        # Line 425-429: Get effective caption lengths
        if cap_mask is not None:
            l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        else:
            l_effective_cap_len = [num_tokens] * bsz

        if cap_mask is not None and not torch.is_floating_point(cap_mask):
            cap_mask = (cap_mask - 1).to(dtype) * torch.finfo(dtype).max

        # Line 431-432: Get image sizes and lengths
        img_sizes = [(img.size(1), img.size(2)) for img in x]
        l_effective_img_len = [(H // pH) * (W // pW) for (H, W) in img_sizes]

        # Line 434-440: Calculate max sequence lengths
        max_seq_len = max(
            (cap_len + img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len))
        )
        max_cap_len = max(l_effective_cap_len)
        max_img_len = max(l_effective_img_len)

        # Line 442: Build position IDs
        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)

        # Line 444-456: Fill position IDs
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            H, W = img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW
            assert H_tokens * W_tokens == img_len

            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:cap_len+img_len, 0] = cap_len
            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            position_ids[i, cap_len:cap_len+img_len, 1] = row_ids
            position_ids[i, cap_len:cap_len+img_len, 2] = col_ids

        # Line 458: Generate RoPE embeddings
        # rope_embedder returns: [batch, seq_len, rope_dim, 2, 2]
        # After movedim(1,2): [batch, rope_dim, seq_len, 2, 2]  
        # BUT WAIT - the original code treats it as [batch, seq, ...] so dim1 is sequence!
        freqs_cis = self.rope_embedder(position_ids).movedim(1, 2).to(dtype)
        
        logging.debug(f"[ONNX] freqs_cis shape: {freqs_cis.shape}")

        # Line 460-467: Build separate freqs for caption and image
        # CRITICAL: Original code uses dimension 1 for sequence, NOT dimension 2!
        cap_freqs_cis_shape = list(freqs_cis.shape)
        cap_freqs_cis_shape[1] = cap_feats.shape[1]  # dim 1 is seq in original code
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        img_freqs_cis_shape = list(freqs_cis.shape)
        img_freqs_cis_shape[1] = max_img_len  # dim 1 is seq
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        # Line 469-473: Fill by slicing (dimension 1 is sequence!)
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len:cap_len+img_len]

        # Line 475-477: Refine context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        # Line 479-489: Patchify and embed images
        flat_x = []
        for i in range(bsz):
            img = x[i]
            C, H, W = img.size()
            # EXACTLY match original: view + permute + flatten
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
        
        # Line 491-492: Refine image embeddings
        for layer in self.noise_refiner:
            padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t)

        # Line 494-498: Build combined mask
        if cap_mask is not None:
            mask = torch.zeros(bsz, max_seq_len, dtype=dtype, device=device)
            mask[:, :max_cap_len] = cap_mask[:, :max_cap_len]
        else:
            mask = None

        # Line 500-507: Build final embedding
        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=x[0].dtype)
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            padded_full_embed[i, cap_len:cap_len+img_len] = padded_img_embed[i, :img_len]

        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis
    
    # Apply patch
    lumina_model.NextDiT.patchify_and_embed = onnx_compatible_patchify_and_embed
    logging.info("[TensorRT] ✅ Patched Lumina2 NextDiT for ONNX export (source-accurate)")
    
    return True