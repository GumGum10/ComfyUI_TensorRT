"""
VAE Debugging Hook for Lumina2 TensorRT
Helps diagnose vertical line artifacts by comparing PyTorch vs TensorRT latents
"""

import torch
import os
from PIL import Image
import numpy as np


class VAEDebugHook:
    """Hook into VAE decode to analyze latents before decoding"""
    
    def __init__(self, output_dir="output/vae_debug"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sample_count = 0
        
    def analyze_latent(self, latent, source="unknown"):
        """Analyze a latent tensor before VAE decoding"""
        self.sample_count += 1
        
        print(f"\n{'='*80}")
        print(f"[VAE Debug] Analyzing Latent #{self.sample_count} from {source}")
        print(f"{'='*80}")
        
        # Basic properties
        print(f"\n📊 Tensor Properties:")
        print(f"  Shape: {latent.shape}")
        print(f"  Dtype: {latent.dtype}")
        print(f"  Device: {latent.device}")
        print(f"  Is contiguous: {latent.is_contiguous()}")
        print(f"  Stride: {latent.stride()}")
        
        # Statistical analysis
        print(f"\n📈 Statistics:")
        print(f"  Min: {latent.min().item():.6f}")
        print(f"  Max: {latent.max().item():.6f}")
        print(f"  Mean: {latent.mean().item():.6f}")
        print(f"  Std: {latent.std().item():.6f}")
        print(f"  Median: {latent.median().item():.6f}")
        
        # Check for anomalies
        print(f"\n⚠️  Anomaly Check:")
        has_nan = torch.isnan(latent).any().item()
        has_inf = torch.isinf(latent).any().item()
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            print(f"  ❌ CRITICAL: Latent contains NaN or Inf values!")
            return
        
        # Channel-wise analysis (Lumina2 has 16 latent channels)
        print(f"\n📺 Per-Channel Analysis:")
        if len(latent.shape) == 4:  # [B, C, H, W]
            B, C, H, W = latent.shape
            print(f"  Batch: {B}, Channels: {C}, Height: {H}, Width: {W}")
            
            for c in range(min(C, 4)):  # Show first 4 channels
                ch_data = latent[0, c]
                print(f"  Channel {c}: min={ch_data.min().item():.4f}, "
                      f"max={ch_data.max().item():.4f}, "
                      f"mean={ch_data.mean().item():.4f}, "
                      f"std={ch_data.std().item():.4f}")
        
        # Spatial pattern analysis (detect vertical/horizontal anomalies)
        print(f"\n🔍 Spatial Pattern Analysis:")
        if len(latent.shape) == 4:
            # Check for vertical patterns (column-wise variance)
            col_variance = latent[0].var(dim=(0, 1))  # Variance across channels and height
            row_variance = latent[0].var(dim=(0, 2))  # Variance across channels and width
            
            print(f"  Column variance: min={col_variance.min().item():.6f}, "
                  f"max={col_variance.max().item():.6f}, "
                  f"mean={col_variance.mean().item():.6f}")
            print(f"  Row variance: min={row_variance.min().item():.6f}, "
                  f"max={row_variance.max().item():.6f}, "
                  f"mean={row_variance.mean().item():.6f}")
            
            # Detect if there are suspicious patterns
            col_var_ratio = col_variance.max() / (col_variance.mean() + 1e-8)
            row_var_ratio = row_variance.max() / (row_variance.mean() + 1e-8)
            
            if col_var_ratio > 5.0:
                print(f"  ⚠️  WARNING: High column variance ratio ({col_var_ratio:.2f}) - may cause vertical artifacts")
            if row_var_ratio > 5.0:
                print(f"  ⚠️  WARNING: High row variance ratio ({row_var_ratio:.2f}) - may cause horizontal artifacts")
        
        # Save visualization
        self._save_latent_visualization(latent, source)
        
        print(f"{'='*80}\n")
    
    def _save_latent_visualization(self, latent, source):
        """Save latent channels as images for visual inspection"""
        if len(latent.shape) != 4:
            return
        
        try:
            B, C, H, W = latent.shape
            
            # Save first 4 channels as images
            for c in range(min(C, 4)):
                ch_data = latent[0, c].detach().cpu().float()
                
                # Normalize to 0-255
                ch_min = ch_data.min()
                ch_max = ch_data.max()
                if ch_max > ch_min:
                    ch_normalized = ((ch_data - ch_min) / (ch_max - ch_min) * 255).numpy().astype(np.uint8)
                else:
                    ch_normalized = np.zeros_like(ch_data.numpy(), dtype=np.uint8)
                
                # Save as PNG
                img = Image.fromarray(ch_normalized, mode='L')
                filename = f"{self.output_dir}/latent_{self.sample_count:03d}_{source}_ch{c:02d}.png"
                img.save(filename)
            
            print(f"  💾 Saved latent visualizations to: {self.output_dir}/")
        except Exception as e:
            print(f"  ⚠️  Failed to save visualization: {e}")
    
    def compare_latents(self, latent_trt, latent_pytorch):
        """Compare TensorRT latent vs PyTorch latent"""
        print(f"\n{'='*80}")
        print(f"[VAE Debug] Comparing TensorRT vs PyTorch Latents")
        print(f"{'='*80}")
        
        if latent_trt.shape != latent_pytorch.shape:
            print(f"  ❌ ERROR: Shape mismatch!")
            print(f"     TensorRT: {latent_trt.shape}")
            print(f"     PyTorch: {latent_pytorch.shape}")
            return
        
        # Compute differences
        diff = (latent_trt - latent_pytorch).abs()
        rel_diff = diff / (latent_pytorch.abs() + 1e-8)
        
        print(f"\n📊 Absolute Difference:")
        print(f"  Min: {diff.min().item():.6f}")
        print(f"  Max: {diff.max().item():.6f}")
        print(f"  Mean: {diff.mean().item():.6f}")
        print(f"  Std: {diff.std().item():.6f}")
        
        print(f"\n📊 Relative Difference (%):")
        print(f"  Min: {(rel_diff.min().item() * 100):.2f}%")
        print(f"  Max: {(rel_diff.max().item() * 100):.2f}%")
        print(f"  Mean: {(rel_diff.mean().item() * 100):.2f}%")
        
        # Check correlation
        trt_flat = latent_trt.flatten()
        pt_flat = latent_pytorch.flatten()
        correlation = torch.corrcoef(torch.stack([trt_flat, pt_flat]))[0, 1]
        print(f"\n📊 Correlation: {correlation.item():.6f}")
        
        if correlation < 0.95:
            print(f"  ⚠️  WARNING: Low correlation - TensorRT output differs significantly!")
        
        print(f"{'='*80}\n")


# Global debug hook instance
_debug_hook = None

def enable_vae_debug():
    """Enable VAE debugging"""
    global _debug_hook
    _debug_hook = VAEDebugHook()
    print("[VAE Debug] Debugging enabled!")
    return _debug_hook

def get_vae_debug_hook():
    """Get the current debug hook"""
    return _debug_hook
