"""
Enhanced TensorRT logging for detailed build progress visibility
"""
import tensorrt as trt
import time
import logging
import re


class DetailedTRTLogger(trt.ILogger):
    """
    Custom TensorRT logger that exposes detailed kernel profiling and tactic selection
    """
    def __init__(self):
        trt.ILogger.__init__(self)
        self.kernel_count = 0
        self.tactic_count = 0
        self.layer_timings = []
        self.current_layer = None
        self.start_time = time.time()
        self.last_progress_time = time.time()
        
    def log(self, severity, msg):
        now = time.time()
        elapsed = now - self.start_time
        
        # Parse different message types
        
        # 1. Layer compilation start
        if "Compiling" in msg or "Building" in msg:
            match = re.search(r'layer (\d+)', msg, re.IGNORECASE)
            if match:
                layer_num = match.group(1)
                self.current_layer = layer_num
                print(f"[{elapsed:.1f}s] 🔨 Compiling layer {layer_num}: {msg[:80]}")
        
        # 2. Tactic selection
        elif "tactic" in msg.lower() or "tactics" in msg.lower():
            self.tactic_count += 1
            if "selected" in msg.lower() or "fastest" in msg.lower():
                print(f"[{elapsed:.1f}s] ✅ Tactic selected: {msg[:100]}")
            elif self.tactic_count % 10 == 0:  # Show every 10th tactic tested
                print(f"[{elapsed:.1f}s] 🧪 Testing tactics ({self.tactic_count} tested): {msg[:80]}")
        
        # 3. Kernel profiling
        elif "kernel" in msg.lower():
            self.kernel_count += 1
            if "ms" in msg or "time" in msg.lower():
                # Extract timing if present
                timing_match = re.search(r'(\d+\.?\d*)\s*ms', msg)
                if timing_match:
                    kernel_time = timing_match.group(1)
                    print(f"[{elapsed:.1f}s] ⚡ Kernel {self.kernel_count}: {kernel_time}ms - {msg[:70]}")
                elif self.kernel_count % 50 == 0:
                    print(f"[{elapsed:.1f}s] ⚙️  Profiled {self.kernel_count} kernels so far...")
        
        # 4. Optimization passes
        elif "optimization" in msg.lower() or "optimizing" in msg.lower():
            print(f"[{elapsed:.1f}s] 🔧 {msg[:100]}")
        
        # 5. Layer fusion
        elif "fus" in msg.lower():
            print(f"[{elapsed:.1f}s] 🔗 Layer fusion: {msg[:80]}")
        
        # 6. Cost estimation
        elif "cost" in msg.lower() and ("compute" in msg.lower() or "estimate" in msg.lower()):
            print(f"[{elapsed:.1f}s] 💰 Cost estimation: {msg[:80]}")
        
        # 7. Progress indicators (percentage, ratios, etc)
        elif any(indicator in msg for indicator in ['%', '/', 'progress']):
            # Throttle these to every 5 seconds
            if (now - self.last_progress_time) > 5.0:
                print(f"[{elapsed:.1f}s] 📊 {msg[:100]}")
                self.last_progress_time = now
        
        # 8. Warnings and Errors (always show)
        elif severity <= trt.Logger.WARNING:
            level = "⚠️  WARNING" if severity == trt.Logger.WARNING else "❌ ERROR"
            print(f"[{elapsed:.1f}s] {level}: {msg}")
        
        # 9. Important info messages
        elif severity <= trt.Logger.INFO:
            if any(keyword in msg.lower() for keyword in [
                'engine', 'network', 'profile', 'builder', 
                'serializ', 'finish', 'complete', 'success'
            ]):
                print(f"[{elapsed:.1f}s] ℹ️  {msg}")
    
    def get_stats(self):
        """Return statistics about the build process"""
        return {
            'kernels_profiled': self.kernel_count,
            'tactics_tested': self.tactic_count,
            'elapsed_time': time.time() - self.start_time
        }


def create_verbose_logger():
    """Factory function to create the detailed logger"""
    return DetailedTRTLogger()