"""
Quick script to enable VAE debugging for Lumina2 TensorRT
Run this in ComfyUI's Python console or add to __init__.py
"""

from .vae_debug_hook import enable_vae_debug

# Enable debugging
enable_vae_debug()
print("✅ VAE Debug hook enabled! Spatial pattern analysis will run on next inference.")
