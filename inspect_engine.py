"""
TensorRT Engine Inspector
Helps verify if your Lumina2 engine was built with the artifact fix
"""
import tensorrt as trt
import os
import sys

def inspect_engine(engine_path):
    """Inspect a TensorRT engine file"""
    if not os.path.exists(engine_path):
        print(f"❌ Engine file not found: {engine_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"🔍 Inspecting TensorRT Engine")
    print(f"{'='*80}")
    print(f"File: {engine_path}")
    print(f"Size: {os.path.getsize(engine_path) / (1024**3):.2f} GB")
    print(f"Modified: {os.path.getmtime(engine_path)}")
    
    # Load engine
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    print(f"\n📊 Engine Info:")
    print(f"  Num I/O tensors: {engine.num_io_tensors}")
    print(f"  Num optimization profiles: {engine.num_optimization_profiles}")
    
    print(f"\n📥 Input Tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            print(f"  {name}: {list(shape)} ({dtype})")
            
            # Check for dynamic shapes
            for profile in range(engine.num_optimization_profiles):
                min_shape = engine.get_tensor_profile_shape(name, profile)[0]
                opt_shape = engine.get_tensor_profile_shape(name, profile)[1]
                max_shape = engine.get_tensor_profile_shape(name, profile)[2]
                print(f"    Profile {profile}: min={list(min_shape)}, opt={list(opt_shape)}, max={list(max_shape)}")
    
    print(f"\n📤 Output Tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            print(f"  {name}: {list(shape)} ({dtype})")
    
    # Try to determine if this is Lumina2
    has_context_input = any(
        "context" in engine.get_tensor_name(i).lower() 
        for i in range(engine.num_io_tensors)
    )
    
    print(f"\n🎯 Model Detection:")
    if has_context_input:
        print(f"  ✅ Likely a Lumina2 model (has 'context' input)")
        print(f"\n⚠️  IMPORTANT:")
        print(f"  If this engine was built BEFORE the unpatchify fix,")
        print(f"  you will see vertical line artifacts in generated images.")
        print(f"  ")
        print(f"  To check if it needs rebuilding:")
        print(f"  1. Check the 'Modified' timestamp above")
        print(f"  2. Compare with when you applied the patch")
        print(f"  3. If older than the patch, DELETE and REBUILD")
    else:
        print(f"  ℹ️  Not a Lumina2 model (no 'context' input)")
    
    print(f"\n{'='*80}\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_engine.py <path_to_engine>")
        print("\nExample:")
        print("  python inspect_engine.py output/tensorrt/lumina2_model.engine")
        sys.exit(1)
    
    engine_path = sys.argv[1]
    inspect_engine(engine_path)
