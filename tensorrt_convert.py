import torch
import sys
import os
import time
import logging

def setup_tensorrt_logging(log_file="tensorrt_build.log"):
    """Setup comprehensive logging to file"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (everything)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Also capture print statements
    class TeeOutput:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, 'a', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Redirect stdout to also write to file
    sys.stdout = TeeOutput(log_file)
    
    return log_file


import threading
import subprocess
import shutil

# CRITICAL: Set TensorRT logger BEFORE importing tensorrt!
import ctypes
import sys

class PreInitTRTLogger:
    """Set logger before TensorRT loads"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.messages = []
        
    def log(self, severity, msg):
        timestamp = time.strftime('%H:%M:%S')
        self.messages.append((timestamp, severity, msg))
        # Print immediately
        if "tactic" in msg.lower() or "kernel" in msg.lower():
            print(f"[{timestamp}] 🔍 {msg}")
        elif "error" in msg.lower():
            print(f"[{timestamp}] ❌ {msg}")
        elif "warning" in msg.lower():
            print(f"[{timestamp}] ⚠️  {msg}")

# Pre-initialize logger
_pre_logger = PreInitTRTLogger()

# NOW import TensorRT
import tensorrt as trt

# Override the logger immediately
try:
    # This must happen before any Builder/Runtime creation
    original_logger = trt.Logger(trt.Logger.VERBOSE)
except Exception as e:
    print(f"Failed to override logger: {e}")

import comfy.model_management
import folder_paths
from tqdm import tqdm

try:
    from .tensorrt_verbose_logging import create_verbose_logger
    VERBOSE_LOGGING_AVAILABLE = True
except ImportError:
    VERBOSE_LOGGING_AVAILABLE = False
    logging.warning("[TensorRT] Verbose logging not available")

try:
    from .lumina2_onnx_patch import patch_lumina2_for_onnx_export
    LUMINA2_PATCH_AVAILABLE = True
except ImportError:
    LUMINA2_PATCH_AVAILABLE = False
    logging.warning("[TensorRT] Lumina2 ONNX patch not available")


class GPUProbe:
    """Lightweight GPU telemetry using NVML if available, else nvidia-smi fallback.
    Exposes a stats_str() with utilization, memory, temp, and power.
    """

    def __init__(self, device_index: int | None = None):
        self.idx = int(device_index) if device_index is not None else (torch.cuda.current_device() if torch.cuda.is_available() else 0)
        self._nvml = None
        self._nvml_device = None
        # Try NVML first
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_device = pynvml.nvmlDeviceGetHandleByIndex(self.idx)
        except Exception:
            # Fallback to nvidia-smi via subprocess
            self._nvml = None
            self._nvml_device = None

    def _stats_via_nvml(self) -> str | None:
        try:
            if self._nvml is None or self._nvml_device is None:
                return None
            nvml = self._nvml
            h = self._nvml_device
            util = nvml.nvmlDeviceGetUtilizationRates(h)
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            temp = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
            try:
                power = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # W
            except Exception:
                power = None
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            pwr = f" {power:.0f}W" if power is not None else ""
            return f"GPU{self.idx} {util.gpu}% {used_gb:.1f}/{total_gb:.1f} GB {temp}C{pwr}"
        except Exception:
            return None

    def _stats_via_nvsmi(self) -> str | None:
        try:
            if shutil.which("nvidia-smi") is None:
                return None
            # Query a single GPU index
            q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
            cmd = [
                "nvidia-smi",
                f"--id={self.idx}",
                f"--query-gpu={q}",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=1.0)
            # Example: "25, 1550, 24564, 45, 120.50"
            parts = [p.strip() for p in out.strip().split(",")]
            if len(parts) >= 5:
                util = float(parts[0])
                used_mb = float(parts[1])
                total_mb = float(parts[2])
                temp = float(parts[3])
                power = float(parts[4])
                used_gb = used_mb / 1024.0
                total_gb = total_mb / 1024.0
                return f"GPU{self.idx} {util:.0f}% {used_gb:.1f}/{total_gb:.1f} GB {temp:.0f}C {power:.0f}W"
            return None
        except Exception:
            return None

    def stats_str(self) -> str:
        # Prefer NVML; fallback to nvidia-smi; else minimal torch memory
        s = self._stats_via_nvml()
        if s:
            return s
        s = self._stats_via_nvsmi()
        if s:
            return s
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = total - free
                return f"GPU{self.idx} {used/1e9:.1f}/{total/1e9:.1f} GB"
        except Exception:
            pass
        return "GPU N/A"


# add output directory to tensorrt search path
if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.get_output_directory(), "tensorrt")
    )
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.get_output_directory(), "tensorrt")],
        {".engine"},
    )

class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self, gpu_probe: GPUProbe | None = None):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5
        self._last_log_time = {}
        self._phase_meta = {}  # phase_name -> {total, start_time}
        self._log_interval = 2.0  # seconds
        self._gpu_probe = gpu_probe

    def _gpu_mem_str(self):
        try:
            if self._gpu_probe is not None:
                return self._gpu_probe.stats_str()
        except Exception:
            pass
        # Fallback minimal
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = total - free
                return f"GPU {used/1e9:.1f}/{total/1e9:.1f} GB"
        except Exception:
            pass
        return "GPU N/A"

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
            self._phase_meta[phase_name] = {"total": int(num_steps), "start_time": time.time()}
            self._last_log_time[phase_name] = 0.0
            logging.info(f"[TRT][phase_start] {phase_name}: total_steps={num_steps}")
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
                meta = self._phase_meta.pop(phase_name, {"start_time": time.time(), "total": 0})
                elapsed = time.time() - meta.get("start_time", time.time())
                logging.info(f"[TRT][phase_finish] {phase_name}: elapsed={elapsed:.1f}s")
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
                # Throttled progress log
                now = time.time()
                last = self._last_log_time.get(phase_name, 0.0)
                total = max(1, self._phase_meta.get(phase_name, {}).get("total", 1))
                if (now - last) >= self._log_interval or step >= (total - 1):
                    self._last_log_time[phase_name] = now
                    start_time = self._phase_meta.get(phase_name, {}).get("start_time", now)
                    elapsed = now - start_time
                    percent = 100.0 * min(max(step, 0), total) / total
                    speed = step / elapsed if elapsed > 0 else 0.0
                    remaining = (total - step) / speed if speed > 0 else float("inf")
                    logging.info(
                        f"[TRT][progress] {phase_name}: {step}/{total} ({percent:.1f}%) "
                        f"elapsed={elapsed:.1f}s eta={(remaining if remaining!=float('inf') else 0):.1f}s {self._gpu_mem_str()}"
                    )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False
        

class TRT_MODEL_CONVERSION_BASE:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        # Default timing cache path; will be customized per model prefix at runtime
        self._timing_cache_dir = os.path.dirname(os.path.realpath(__file__))
        self._default_timing_cache_path = os.path.normpath(
            os.path.join(self._timing_cache_dir, "timing_cache.trt")
        )
        self._current_timing_cache_path = self._default_timing_cache_path

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(s):
        raise NotImplementedError

    def _update_timing_cache_path(self, filename_prefix: str):
        """Create a per-prefix timing cache to speed up repeated builds for the same model/config."""
        try:
            slug = (
                filename_prefix.replace("\\", "_")
                .replace("/", "_")
                .replace(":", "_")
                .replace(" ", "_")
            )
            # Keep the filename short but unique enough
            slug = slug[-120:]
            self._current_timing_cache_path = os.path.normpath(
                os.path.join(self._timing_cache_dir, f"timing_cache_{slug}.trt")
            )
        except Exception:
            self._current_timing_cache_path = self._default_timing_cache_path

    # Sets up the builder to use the timing cache file, and creates it if it does not already exist
    def _setup_timing_cache(self, config: trt.IBuilderConfig):
        buffer = b""
        path = self._current_timing_cache_path
        if os.path.exists(path):
            with open(path, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
            print("Read {} bytes from timing cache ({}).".format(len(buffer), os.path.basename(path)))
        else:
            print("No timing cache found; Initializing a new one.")
        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Saves the config's timing cache to file
    def _save_timing_cache(self, config: trt.IBuilderConfig):
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        path = self._current_timing_cache_path
        with open(path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def _start_progress_heartbeat(self, monitor: "TQDMProgressMonitor", interval_s: float = 5.0):
        """Start a background thread that logs a heartbeat for the current active phase.

        Returns a tuple (stop_event, thread). Caller must set stop_event and join thread when done.
        """
        stop_event = threading.Event()

        def _heartbeat():
            while not stop_event.wait(interval_s):
                try:
                    # Find the deepest active phase (max indent)
                    if not monitor._active_phases:
                        continue
                    # Choose the phase with highest nbIndents (deepest) or the last started
                    phase = None
                    max_indent = -1
                    for name, data in list(monitor._active_phases.items()):
                        indent = data.get("nbIndents", 0)
                        if indent >= max_indent:
                            max_indent = indent
                            phase = name
                    if phase is None:
                        continue
                    tq = monitor._active_phases[phase]["tq"]
                    total = max(1, int(tq.total))
                    step = int(tq.n)
                    percent = 100.0 * min(step, total) / total
                    meta = monitor._phase_meta.get(phase, {})
                    start_time = meta.get("start_time", time.time())
                    elapsed = time.time() - start_time
                    logging.info(
                        f"[TRT][heartbeat] {phase}: {step}/{total} ({percent:.1f}%) elapsed={elapsed:.1f}s {monitor._gpu_mem_str()}"
                    )
                except Exception:
                    # Best-effort; ignore heartbeat errors
                    pass

        th = threading.Thread(target=_heartbeat, name="TRTProgressHeartbeat", daemon=True)
        th.start()
        return stop_event, th

    def _convert(
        self,
        model,
        filename_prefix,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        context_min,
        context_opt,
        context_max,
        num_video_frames,
        is_static: bool,
        fast_build: bool = False,
    ):
        tensorrt_log_dir = os.path.join(self.output_dir, "tensorrt", "logs")
        os.makedirs(tensorrt_log_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"tensorrt_build_{timestamp}.log"
        log_file = os.path.join(tensorrt_log_dir, log_filename)
        
        setup_tensorrt_logging(log_file)
        logging.info(f"[TensorRT] ========================================")
        logging.info(f"[TensorRT] Log file: {log_file}")
        logging.info(f"[TensorRT] ========================================")
        print(f"\n📝 TensorRT build log: {log_file}\n")

        # Update per-prefix timing cache path early
        self._update_timing_cache_path(filename_prefix)
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(self.temp_dir, "{}".format(time.time())), "model.onnx"
            )
        )

        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)
        unet = model.model.diffusion_model

        is_lumina2 = isinstance(model.model, comfy.model_base.Lumina2)
        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        context_len = 77
        context_len_min = context_len
        y_dim = model.model.adm_channels
        extra_input = {}
        dtype = torch.float16
        # In tensorrt_convert.py, around line 360, ADD THIS:
        logging.info(f"[TensorRT] ===== MODEL DETECTION =====")
        logging.info(f"[TensorRT] Model type: {type(model.model)}")
        logging.info(f"[TensorRT] Model base: {type(model.model).__bases__}")
        logging.info(f"[TensorRT] Is SD3: {isinstance(model.model, comfy.model_base.SD3)}")
        logging.info(f"[TensorRT] Is Flux: {isinstance(model.model, comfy.model_base.Flux)}")
        logging.info(f"[TensorRT] Is Lumina2: {isinstance(model.model, comfy.model_base.Lumina2)}")
        logging.info(f"[TensorRT] Model config: {model.model.model_config.__class__.__name__}")
        logging.info(f"[TensorRT] =============================")

        if isinstance(model.model, comfy.model_base.SD3): #SD3
            context_embedder_config = model.model.model_config.unet_config.get("context_embedder_config", None)
            if context_embedder_config is not None:
                context_dim = context_embedder_config.get("params", {}).get("in_features", None)
                context_len = 154 #NOTE: SD3 can have 77 or 154 depending on which text encoders are used, this is why context_len_min stays 77
        elif isinstance(model.model, comfy.model_base.AuraFlow):
            context_dim = 2048
            context_len_min = 256
            context_len = 256
        elif isinstance(model.model, comfy.model_base.Flux):
            context_dim = model.model.model_config.unet_config.get("context_in_dim", None)
            context_len_min = 256
            context_len = 256
            y_dim = model.model.model_config.unet_config.get("vec_in_dim", None)
            extra_input = {"guidance": ()}
            dtype = torch.bfloat16
        elif is_lumina2:
            logging.info("[TensorRT] Detected Lumina2 model, configuring NextDiT architecture")
            
            # Apply ONNX compatibility patch
            if LUMINA2_PATCH_AVAILABLE:
                patch_lumina2_for_onnx_export()
            else:
                raise RuntimeError("Lumina2 ONNX patch is required but not available. Please ensure lumina2_onnx_patch.py is in the same directory.")
            
            # Use actual Gemma2-2B embedding dimension
            context_dim = 2304
            context_len_min = 512
            context_len = 512
            
            y_dim = 0
            dtype = torch.bfloat16

           
            # CRITICAL FIX: Don't add num_tokens as a dynamic input
            # Instead, we'll wrap the model to use context.shape[1] as num_tokens
            extra_input = {}  # Changed from {"num_tokens": ()}
            
            logging.info(f"[TensorRT] Lumina2 Configuration:")
            logging.info(f"[TensorRT] - Context Dimension: {context_dim}")
            logging.info(f"[TensorRT] - Context Length Min/Opt: {context_len_min}/{context_len}")
            logging.info(f"[TensorRT] - Data Type: {dtype}")
            
            print(f"✅ Lumina2 NextDiT ready for ONNX export (context_dim={context_dim})")

        if context_dim is not None:
            input_names = ["x", "timesteps", "context"]
            output_names = ["h"]

            dynamic_axes = {
                "x": {0: "batch", 2: "height", 3: "width"},
                "timesteps": {0: "batch"},
                "context": {0: "batch", 1: "seq_len"},  # FIXED: Use seq_len for Lumina2
            }

            transformer_options = model.model_options['transformer_options'].copy()
            
            # LUMINA2: Wrap model now that we have transformer_options
            if is_lumina2:
                class Lumina2ONNXWrapper(torch.nn.Module):
                    """
                    ONNX-compatible wrapper for Lumina2 that derives num_tokens from context shape
                    """
                    def __init__(self, unet, transformer_options):
                        super().__init__()
                        self.unet = unet
                        self.transformer_options = transformer_options
                        
                    def forward(self, x, timesteps, context):
                        """
                        Args:
                            x: [B, C, H, W] latent tensor
                            timesteps: [B] timestep tensor  
                            context: [B, L, D] text embeddings
                        """
                        # Derive num_tokens from context sequence length
                        # This makes it a constant in the ONNX graph based on the input shape
                        num_tokens = context.shape[1]
                        
                        # Call original model
                        return self.unet(
                            x,
                            timesteps,
                            context,
                            num_tokens=num_tokens,
                            attention_mask=None,
                            transformer_options=self.transformer_options
                        )
                
                # Wrap the model
                unet = Lumina2ONNXWrapper(unet, transformer_options)
                logging.info("[TensorRT] Wrapped Lumina2 model with ONNX-compatible wrapper")
            
            if model.model.model_config.unet_config.get(
                "use_temporal_resblock", False
            ):  # SVD
                batch_size_min = num_video_frames * batch_size_min
                batch_size_opt = num_video_frames * batch_size_opt
                batch_size_max = num_video_frames * batch_size_max

                class UNET(torch.nn.Module):
                    def forward(self, x, timesteps, context, y):
                        return self.unet(
                            x,
                            timesteps,
                            context,
                            y,
                            num_video_frames=self.num_video_frames,
                            transformer_options=self.transformer_options,
                        )

                svd_unet = UNET()
                svd_unet.num_video_frames = num_video_frames
                svd_unet.unet = unet
                svd_unet.transformer_options = transformer_options
                unet = svd_unet
                context_len_min = context_len = 1
            elif not is_lumina2:  # Don't wrap again if already wrapped for Lumina2
                class UNET(torch.nn.Module):
                    def forward(self, x, timesteps, context, *args):
                        extras = input_names[3:]
                        extra_args = {}
                        for i in range(len(extras)):
                            extra_args[extras[i]] = args[i]
                        return self.unet(x, timesteps, context, transformer_options=self.transformer_options, **extra_args)

                _unet = UNET()
                _unet.unet = unet
                _unet.transformer_options = transformer_options
                unet = _unet

            input_channels = model.model.model_config.unet_config.get("in_channels", 4)

            inputs_shapes_min = (
                (batch_size_min, input_channels, height_min // 8, width_min // 8),
                (batch_size_min,),
                (batch_size_min, context_len_min * context_min, context_dim),
            )
            inputs_shapes_opt = (
                (batch_size_opt, input_channels, height_opt // 8, width_opt // 8),
                (batch_size_opt,),
                (batch_size_opt, context_len * context_opt, context_dim),
            )
            inputs_shapes_max = (
                (batch_size_max, input_channels, height_max // 8, width_max // 8),
                (batch_size_max,),
                (batch_size_max, context_len * context_max, context_dim),
            )

            if y_dim > 0:
                input_names.append("y")
                dynamic_axes["y"] = {0: "batch"}
                inputs_shapes_min += ((batch_size_min, y_dim),)
                inputs_shapes_opt += ((batch_size_opt, y_dim),)
                inputs_shapes_max += ((batch_size_max, y_dim),)

            for k in extra_input:
                input_names.append(k)
                dynamic_axes[k] = {0: "batch"}
                inputs_shapes_min += ((batch_size_min,) + extra_input[k],)
                inputs_shapes_opt += ((batch_size_opt,) + extra_input[k],)
                inputs_shapes_max += ((batch_size_max,) + extra_input[k],)


            inputs = ()
            for i, shape in enumerate(inputs_shapes_opt):
                # Use float32 for timesteps for better ONNX/TRT compatibility
                in_dtype = torch.float32 if (i == 1) else dtype
                # FIXED: num_tokens should be int32
                if i < len(input_names) and input_names[i] == "num_tokens":
                    in_dtype = torch.int32
                inputs += (
                    torch.zeros(
                        shape,
                        device=comfy.model_management.get_torch_device(),
                        dtype=in_dtype,
                    ),
                )

        else:
            print("ERROR: model not supported.")
            return ()

        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        
        logging.info("[TensorRT] Preparing for ONNX export:")
        logging.info(f"[TensorRT] - Output path: {output_onnx}")
        logging.info(f"[TensorRT] - Input names: {input_names}")
        logging.info(f"[TensorRT] - Output names: {output_names}")
        logging.info(f"[TensorRT] - Dynamic axes: {dynamic_axes}")
        
        # Log input tensor information
        logging.info("[TensorRT] Input tensor details:")
        for idx, (name, tensor) in enumerate(zip(input_names, inputs)):
            logging.info(f"[TensorRT] - {name}:")
            logging.info(f"    Shape: {tensor.shape}")
            logging.info(f"    Dtype: {tensor.dtype}")
            logging.info(f"    Device: {tensor.device}")
            
            # Check for NaN/Inf values
            with torch.no_grad():
                has_nan = tensor.isnan().any().item() if tensor.dtype.is_floating_point else False
                has_inf = tensor.isinf().any().item() if tensor.dtype.is_floating_point else False
                if has_nan or has_inf:
                    logging.warning(f"[TensorRT] Found NaN/Inf in {name} tensor!")
        
        try:
            logging.info("[TensorRT] Starting ONNX export...")
            torch.onnx.export(
                unet,
                inputs,
                output_onnx,
                verbose=False,  # Set to True for debugging
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=dynamic_axes,
            )
            logging.info("[TensorRT] ONNX export completed successfully")
            try:
                file_size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
                logging.info(f"[TensorRT] ONNX file size: {file_size_mb:.1f} MB")
            except Exception:
                pass
            
            # Verify the exported model
            import onnx
            logging.info("[TensorRT] Verifying exported ONNX model...")
            try:
                # Prefer path-based check to handle large external data models
                onnx.checker.check_model(output_onnx)
                logging.info("[TensorRT] ONNX model verification passed (path-based)")

                # Best-effort lightweight introspection without loading external data
                try:
                    onnx_model_light = onnx.load_model(output_onnx, load_external_data=False)  # type: ignore[call-arg]
                    opset_version = onnx_model_light.opset_import[0].version if onnx_model_light.opset_import else 'unknown'
                    logging.info(f"[TensorRT] Model opset version: {opset_version}")
                    graph = onnx_model_light.graph
                    logging.info(f"[TensorRT] Graph structure (light):")
                    logging.info(f" - Nodes: {len(graph.node)}")
                    logging.info(f" - Inputs: {[i.name for i in graph.input]}")
                    logging.info(f" - Outputs: {[o.name for o in graph.output]}")
                except Exception as e_light:
                    logging.debug(f"[TensorRT] Skipping light introspection: {e_light}")

            except Exception as e:
                logging.error(f"[TensorRT] ONNX model verification failed: {str(e)}")
                raise
                
        except Exception as e:
            logging.error(f"[TensorRT] Error during ONNX export: {str(e)}")
            raise

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # TRT conversion starts here
        logging.info("[TensorRT] Starting TensorRT conversion phase")
        if VERBOSE_LOGGING_AVAILABLE:
            logger = create_verbose_logger()
            logging.info("[TensorRT] Using detailed verbose logging")
        else:
            logger = trt.Logger(trt.Logger.VERBOSE)
            
        builder = trt.Builder(logger)
        logging.info(f"[TensorRT] Created TensorRT builder (version: {trt.__version__})")

        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        logging.info("[TensorRT] Created network with explicit batch dimension")
        
        parser = trt.OnnxParser(network, logger)
        logging.info("[TensorRT] Created ONNX parser")
        
        logging.info(f"[TensorRT] Parsing ONNX file: {output_onnx}")
        success = parser.parse_from_file(output_onnx)
        
        # Log any parser errors
        if parser.num_errors > 0:
            logging.error("[TensorRT] Encountered parser errors:")
            for idx in range(parser.num_errors):
                error = parser.get_error(idx)
                logging.error(f"[TensorRT] Parser error {idx + 1}: {error}")
                print(f"Parser error {idx + 1}: {error}")

        if not success:
            logging.error("[TensorRT] ONNX parsing failed!")
            print("ONNX load ERROR")
            return ()
            
        logging.info("[TensorRT] ONNX parsing completed successfully")
        
        # Log network information
        logging.info("[TensorRT] Network information:")
        logging.info(f"[TensorRT] - Number of inputs: {network.num_inputs}")
        logging.info(f"[TensorRT] - Number of outputs: {network.num_outputs}")
        logging.info(f"[TensorRT] - Number of layers: {network.num_layers}")
        
        # Log input tensor information
        logging.info("[TensorRT] Network input details:")
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            logging.info(f"[TensorRT] Input {i}:")
            logging.info(f"  - Name: {tensor.name}")
            logging.info(f"  - Shape: {tensor.shape}")
            logging.info(f"  - Dtype: {tensor.dtype}")
            
        # Log output tensor information
        logging.info("[TensorRT] Network output details:")
        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            logging.info(f"[TensorRT] Output {i}:")
            logging.info(f"  - Name: {tensor.name}")
            logging.info(f"  - Shape: {tensor.shape}")
            logging.info(f"  - Dtype: {tensor.dtype}")

        logging.info("[TensorRT] Creating builder configuration")
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        
        logging.info("[TensorRT] Setting up timing cache")
        self._setup_timing_cache(config)
        # Attach GPU probe to progress monitor for richer telemetry
        gpu_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        config.progress_monitor = TQDMProgressMonitor(gpu_probe=GPUProbe(gpu_idx))
        # Builder configuration: Optimal settings for your hardware
        if fast_build:
            # FAST BUILD MODE (for iteration/testing)
            try:
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 20 << 30)  # 20GB (increased from 4GB)
                logging.info("[TensorRT] [fast_build] Set WORKSPACE to 20 GiB")
            except Exception as e:
                logging.debug(f"[TensorRT] [fast_build] set_memory_pool_limit error: {e}")
            
            try:
                if hasattr(config, 'builder_optimization_level'):
                    config.builder_optimization_level = 0  # DEBUG DEBUG DEBUG 0 -> USES FIRST FOUND ALGO
                    logging.info(f"✨✨✨[TensorRT] [fast_build] builder_optimization_level={config.builder_optimization_level}")
            except Exception as e:
                logging.debug(f"[TensorRT] [fast_build] builder_optimization_level error: {e}")
            
            try:
                if hasattr(trt, 'TacticSource') and hasattr(config, 'set_tactic_sources'):
                    sources = 0
                    # Add more tactic sources for better quality
                    for s in ('CUBLAS', 'CUBLAS_LT'):  # Added CUBLAS
                        if hasattr(trt.TacticSource, s):
                            sources |= getattr(trt.TacticSource, s)
                    if sources:
                        config.set_tactic_sources(sources)
                        logging.info("[TensorRT] [fast_build] Enabled tactic sources: cuBLAS + cuBLASLt")
            except Exception as e:
                logging.debug(f"[TensorRT] [fast_build] set_tactic_sources error: {e}")

        else:
            # PRODUCTION BUILD MODE - OPTIMIZED FOR i7-13700K + 64GB RAM + RTX 4090
            try:
                # MAXIMIZE workspace - 24GB VRAM, 16GB for workspace!
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)  # 16GB (increased from 12GB)
                logging.info("[TensorRT] Set WORKSPACE to 16 GiB (maximizing 4090's 24GB VRAM)")
            except Exception as e:
                logging.debug(f"[TensorRT] set_memory_pool_limit error: {e}")
            
            try:
                # Use Level 3 (NVIDIA recommended) instead of 4 for better time/quality balance
                if hasattr(config, 'builder_optimization_level'):
                    config.builder_optimization_level = 3  # Changed from 4 to 3 (RECOMMENDED)
                    logging.info("[TensorRT] Set builder_optimization_level=3 (NVIDIA recommended)")
            except Exception as e:
                logging.debug(f"[TensorRT] builder_optimization_level error: {e}")
            
            try:
                # Enable ALL tactic sources for maximum GPU utilization
                if hasattr(trt, 'TacticSource') and hasattr(config, 'set_tactic_sources'):
                    sources = 0
                    # Add ALL available tactic sources
                    for s in ('CUBLAS', 'CUBLAS_LT', 'CUDNN', 'EDGE_MASK_CONVOLUTIONS', 'JIT_CONVOLUTIONS'):
                        if hasattr(trt.TacticSource, s):
                            sources |= getattr(trt.TacticSource, s)
                    if sources:
                        config.set_tactic_sources(sources)
                        logging.info("[TensorRT] Enabled ALL tactic sources for maximum GPU utilization")
            except Exception as e:
                logging.debug(f"[TensorRT] set_tactic_sources error: {e}")
            
            # OPTIONAL: Increase timing iterations for more accurate profiling
            try:
                if hasattr(config, 'avg_timing_iterations'):
                    config.avg_timing_iterations = 16  # Default is 8, increase for better accuracy
                    logging.info("[TensorRT] Set avg_timing_iterations=16 (more accurate profiling)")
            except Exception as e:
                logging.debug(f"[TensorRT] avg_timing_iterations error: {e}")
            
            # OPTIONAL: Enable profiling verbosity to see what TRT is doing
            try:
                if hasattr(config, 'profiling_verbosity'):
                    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                    logging.info("[TensorRT] Enabled detailed profiling verbosity")
            except Exception as e:
                logging.debug(f"[TensorRT] profiling_verbosity error: {e}")

        prefix_encode = ""
        for k in range(len(input_names)):
            min_shape = inputs_shapes_min[k]
            opt_shape = inputs_shapes_opt[k]
            max_shape = inputs_shapes_max[k]
            
            logging.info(f"[TensorRT] Setting shape for {input_names[k]}:")
            logging.info(f"  - Minimum shape: {min_shape}")
            logging.info(f"  - Optimal shape: {opt_shape}")
            logging.info(f"  - Maximum shape: {max_shape}")
            
            try:
                profile.set_shape(input_names[k], min_shape, opt_shape, max_shape)
                logging.info(f"[TensorRT] Successfully set shape profile for {input_names[k]}")
            except Exception as e:
                logging.error(f"[TensorRT] Error setting shape for {input_names[k]}: {str(e)}")
                raise

            # Encode shapes to filename
            encode = lambda a: ".".join(map(lambda x: str(x), a))
            prefix_encode += "{}#{}#{}#{};".format(
                input_names[k], encode(min_shape), encode(opt_shape), encode(max_shape)
            )

        if dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        if dtype == torch.bfloat16:
            try:
                config.set_flag(trt.BuilderFlag.BF16)
            except Exception as e:
                logging.warning(f"[TensorRT] BF16 not supported, falling back to FP16: {e}")
                config.set_flag(trt.BuilderFlag.FP16)

        config.add_optimization_profile(profile)

        if is_static:
            filename_prefix = "{}_${}".format(
                filename_prefix,
                "-".join(
                    (
                        "stat",
                        "b",
                        str(batch_size_opt),
                        "h",
                        str(height_opt),
                        "w",
                        str(width_opt),
                    )
                ),
            )
        else:
            filename_prefix = "{}_${}".format(
                filename_prefix,
                "-".join(
                    (
                        "dyn",
                        "b",
                        str(batch_size_min),
                        str(batch_size_max),
                        str(batch_size_opt),
                        "h",
                        str(height_min),
                        str(height_max),
                        str(height_opt),
                        "w",
                        str(width_min),
                        str(width_max),
                        str(width_opt),
                    )
                ),
            )

        logging.info("[TensorRT] Building TensorRT engine")
        try:
            # Start heartbeat before build to show logs inside long steps
            monitor = config.progress_monitor  # type: ignore[attr-defined]
            stop_event = None
            hb_thread = None
            if isinstance(monitor, TQDMProgressMonitor):
                try:
                    stop_event, hb_thread = self._start_progress_heartbeat(monitor, interval_s=5.0)
                except Exception:
                    pass

            serialized_engine = builder.build_serialized_network(network, config)
            logging.info("[TensorRT] Successfully built TensorRT engine")
            
            if serialized_engine is None:
                logging.error("[TensorRT] Engine serialization failed!")
                return ()

            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            )
            output_trt_engine = os.path.join(
                full_output_folder, f"{filename}_{counter:05}_.engine"
            )
            
            logging.info(f"[TensorRT] Saving engine to: {output_trt_engine}")
            with open(output_trt_engine, "wb") as f:
                f.write(serialized_engine)
            logging.info("[TensorRT] Engine saved successfully")

            logging.info("[TensorRT] Saving timing cache")
            self._save_timing_cache(config)
            logging.info("[TensorRT] Timing cache saved")
            
        except Exception as e:
            logging.error(f"[TensorRT] Error during engine building/saving: {str(e)}")
            raise
        finally:
            # Stop heartbeat if it was started
            try:
                if 'stop_event' in locals() and stop_event is not None:
                    stop_event.set()
                if 'hb_thread' in locals() and hb_thread is not None:
                    hb_thread.join(timeout=1.0)
            except Exception:
                pass

        return ()


class DYNAMIC_TRT_MODEL_CONVERSION(TRT_MODEL_CONVERSION_BASE):
    def __init__(self):
        super(DYNAMIC_TRT_MODEL_CONVERSION, self).__init__()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_DYN"}),
                "fast_build": ("BOOLEAN", {"default": False}),
                "batch_size_min": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_max": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_max": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_max": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "context_min": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "context_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "context_max": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "num_video_frames": (
                    "INT",
                    {
                        "default": 14,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        model,
        filename_prefix,
        fast_build,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        context_min,
        context_opt,
        context_max,
        num_video_frames,
    ):
        return super()._convert(
            model,
            filename_prefix,
            batch_size_min,
            batch_size_opt,
            batch_size_max,
            height_min,
            height_opt,
            height_max,
            width_min,
            width_opt,
            width_max,
            context_min,
            context_opt,
            context_max,
            num_video_frames,
            is_static=False,
            fast_build=fast_build,
        )


class STATIC_TRT_MODEL_CONVERSION(TRT_MODEL_CONVERSION_BASE):
    def __init__(self):
        super(STATIC_TRT_MODEL_CONVERSION, self).__init__()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_STAT"}),
                "fast_build": ("BOOLEAN", {"default": True}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "context_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "num_video_frames": (
                    "INT",
                    {
                        "default": 14,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        model,
        filename_prefix,
        fast_build,
        batch_size_opt,
        height_opt,
        width_opt,
        context_opt,
        num_video_frames,
    ):
        return super()._convert(
            model,
            filename_prefix,
            batch_size_opt,
            batch_size_opt,
            batch_size_opt,
            height_opt,
            height_opt,
            height_opt,
            width_opt,
            width_opt,
            width_opt,
            context_opt,
            context_opt,
            context_opt,
            num_video_frames,
            is_static=True,
            fast_build=fast_build,
        )


NODE_CLASS_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION,
    "STATIC_TRT_MODEL_CONVERSION": STATIC_TRT_MODEL_CONVERSION,
}