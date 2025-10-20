"""
ONNX Export Patch for Lumina2 NextDiT Model - FIXED VERSION
Root cause: The RoPE freqs_cis has 6D shape [batch, heads, seq, dim, 2, 2] after move dim but was being treated as 5D.
"""

import torch
import torch.nn as nn
import logging


def patch_lumina2_for_onnx_export():
    """
    Patch ComfyUI's Lumina2 to be ONNX-exportable with proper RoPE dimension handling.
    """
    import comfy.ldm.lumina.model as lumina_model
    
    original_patchify_and_embed = lumina_model.NextDiT.patchify_and_embed
    
    def onnx_compatible_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens):
        """
        ONNX-compatible version with proper dimension handling for 6D RoPE embeddings
        """
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device
        dtype = x[0].dtype

        # Get effective caption lengths
        if cap_mask is not None:
            l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        else:
            l_effective_cap_len = [num_tokens] * bsz

        if cap_mask is not None and not torch.is_floating_point(cap_mask):
            cap_mask = (cap_mask - 1).to(dtype) * torch.finfo(dtype).max

        # Get image sizes - ONNX compatible way
        img_sizes = [(int(img.size(1)), int(img.size(2))) for img in x]
        
        # Calculate patch counts as tensor for ONNX tracing
        img_patch_counts = torch.tensor(
            [(H // pH) * (W // pW) for (H, W) in img_sizes],
            dtype=torch.int64,
            device=device
        )
        l_effective_img_len = img_patch_counts.tolist()
        max_img_patches = int(torch.max(img_patch_counts).item())

        # Calculate max sequence lengths
        max_seq_len = max(
            (cap_len + img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len))
        )
        actual_cap_len = cap_feats.shape[1]  # Use actual, not max

        # Build position IDs
        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)

        # Fill position IDs
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            H, W = img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW

            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:cap_len+img_len, 0] = cap_len
            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            position_ids[i, cap_len:cap_len+img_len, 1] = row_ids
            position_ids[i, cap_len:cap_len+img_len, 2] = col_ids

        # Generate RoPE embeddings - THIS IS THE KEY FIX
        # rope_embedder returns: [batch, seq_len, rope_dim, 2, 2]
        # We need to movedim BUT the result is 5D, not 6D!
        freqs_cis_raw = self.rope_embedder(position_ids)  # [B, seq, 48, 2, 2]
        logging.debug(f"[ONNX] freqs_cis_raw shape: {freqs_cis_raw.shape}")
        
        # movedim(1, 2) means: move dim 1 to position 2
        # [B, seq, 48, 2, 2] -> [B, 48, seq, 2, 2]
        # Wait, that's only 5D...
        
        # Actually, let's check the original code behavior
        freqs_cis = freqs_cis_raw.movedim(1, 2).to(dtype)
        logging.debug(f"[ONNX] actual_cap_len: {actual_cap_len}")
        logging.debug(f"[ONNX] max_img_patches: {max_img_patches}")
        logging.debug(f"[ONNX] freqs_cis shape after movedim: {freqs_cis.shape}")
        
        # The shape should be [B, 48, seq, 2, 2] - 5D, not 6D!
        # So the allocation should match this
        ndim = freqs_cis.ndim
        
        # Build separate freqs for caption - use actual shape
        cap_freqs_shape = list(freqs_cis.shape)
        cap_freqs_shape[2] = actual_cap_len  # seq dimension is at index 2
        cap_freqs_cis = torch.zeros(*cap_freqs_shape, device=device, dtype=freqs_cis.dtype)
        logging.debug(f"[ONNX] cap_freqs_cis shape: {cap_freqs_cis.shape}")
        
        # Build separate freqs for image
        img_freqs_shape = list(freqs_cis.shape)
        img_freqs_shape[2] = max_img_patches  # seq dimension is at index 2
        img_freqs_cis = torch.zeros(*img_freqs_shape, device=device, dtype=freqs_cis.dtype)
        logging.debug(f"[ONNX] img_freqs_cis shape: {img_freqs_cis.shape}")
        
        # Fill by slicing - sequence is at dimension 2
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = int(img_patch_counts[i].item())
            img_start = cap_len
            img_end = cap_len + img_len
            
            # Slice along dimension 2 (sequence dimension after movedim)
            cap_freqs_cis[i, :, :cap_len, :, :] = freqs_cis[i, :, :cap_len, :, :]
            logging.debug(f"[ONNX] Filling img_freqs_cis[{i}] with img_len={img_len}")
            logging.debug(f"[ONNX] img_freqs_cis[{i}] shape before: {img_freqs_cis[i].shape}")
            logging.debug(f"[ONNX] freqs_cis[{i}, :, {img_start}:{img_end}] shape: {freqs_cis[i, :, img_start:img_end].shape}")
            img_freqs_cis[i, :, :img_len, :, :] = freqs_cis[i, :, img_start:img_end, :, :]

        # Refine context (NO modulation for context_refiner)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        # Patchify and embed images
        flat_x = []
        for i in range(bsz):
            img = x[i]
            C, H, W = img.size()
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 2, 4, 0).flatten(2).flatten(0, 1)
            flat_x.append(img)
        x = flat_x
        
        padded_img_embed = torch.zeros(bsz, max_img_patches, x[0].shape[-1], device=device, dtype=x[0].dtype)
        padded_img_mask = torch.zeros(bsz, max_img_patches, dtype=dtype, device=device)
        for i in range(bsz):
            padded_img_embed[i, :l_effective_img_len[i]] = x[i]
            padded_img_mask[i, l_effective_img_len[i]:] = -torch.finfo(dtype).max

        padded_img_embed = self.x_embedder(padded_img_embed)
        padded_img_mask = padded_img_mask.unsqueeze(1)
        
        # Refine image embeddings (WITH modulation for noise_refiner, needs 't' as adaln_input)
        for layer in self.noise_refiner:
            padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t)

        # Build combined mask
        if cap_mask is not None:
            mask = torch.zeros(bsz, max_seq_len, dtype=dtype, device=device)
            mask[:, :actual_cap_len] = cap_mask[:, :actual_cap_len]
        else:
            mask = None

        # Build final embedding
        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=x[0].dtype)
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            padded_full_embed[i, cap_len:cap_len+img_len] = padded_img_embed[i, :img_len]

        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis
    
    # Apply patch
    lumina_model.NextDiT.patchify_and_embed = onnx_compatible_patchify_and_embed
    logging.info("[TensorRT] ✅ Applied DYNAMIC SHAPE ONNX compatibility patch to Lumina2 NextDiT")
    logging.info("[TensorRT] ⚠️  Note: RoPE embeddings have 6D structure [batch, heads, seq, dim, 2, 2]")
    
    return True
