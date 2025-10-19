import torch
import sys
import os
import time
import logging
import comfy.model_management

import tensorrt as trt
import folder_paths
from tqdm import tqdm

# TODO:
# Make it more generic: less model specific code

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
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5
        self._last_log_time = {}
        self._phase_meta = {}  # phase_name -> {total, start_time}
        self._log_interval = 2.0  # seconds

    def _gpu_mem_str(self):
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
            
            # Lumina2-specific configuration based on the paper
            context_dim = 2048  # Fixed context dimension for NextDiT
            text_tokens = 77  # Base text token length
            image_tokens = 256  # Image token length from paper
            context_len_min = text_tokens
            context_len = text_tokens + image_tokens  # Total sequence length
            y_dim = 0  # Lumina2 doesn't use ADM channels
            dtype = torch.bfloat16  # Using bfloat16 as specified
            
            logging.info(f"[TensorRT] Lumina2 Configuration:")
            logging.info(f"[TensorRT] - Context Dimension: {context_dim}")
            logging.info(f"[TensorRT] - Text Tokens: {text_tokens}")
            logging.info(f"[TensorRT] - Image Tokens: {image_tokens}")
            logging.info(f"[TensorRT] - Total Sequence Length: {context_len}")
            logging.info(f"[TensorRT] - Data Type: {dtype}")            
            class NextDiTWrapper(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = None
                    self.debug = True  # Enable debug logging

                def set_unet(self, unet):
                    self.unet = unet
                    return self

                def forward(self, x, timesteps, context, *args, **kwargs):
                    logging.info(f"[NextDiT] Processing input tensors:")
                    logging.info(f"[NextDiT] - Input shape (x): {x.shape}")
                    logging.info(f"[NextDiT] - Timesteps shape: {timesteps.shape}")
                    logging.info(f"[NextDiT] - Context shape (before): {context.shape}")
                    
                    # Handle text and latent tokens for NextDiT architecture
                    if not isinstance(context, torch.Tensor):
                        raise TypeError(f"[NextDiT] Expected context to be a tensor, got {type(context)}")
                    
                    B, L, D = context.shape
                    logging.info(f"[NextDiT] Context dimensions: batch={B}, length={L}, dim={D}")
                    
                    if D != 2304:
                        if D == 2048:
                            # Pad from 2048 to 2304 to match the model's internal layers
                            logging.info("[NextDiT] Padding context from 2048 to 2304 dimensions")
                            pad_size = 2304 - 2048
                            context = torch.nn.functional.pad(
                                context, 
                                (0, pad_size), 
                                mode='constant', 
                                value=0
                            ).contiguous()
                            logging.info(f"[NextDiT] Context padded shape: {context.shape}")
                        else:
                            raise ValueError(f"[NextDiT] Unexpected context dimension: {D} (expected 2048 or 2304)")
                    
                    if context.isnan().any():
                        logging.error("[NextDiT] Found NaN values in context!")
                        raise ValueError("[NextDiT] Context tensor contains NaN values")
                    
                    # Log tensor statistics
                    with torch.no_grad():
                        ctx_mean = context.mean().item()
                        ctx_std = context.std().item()
                        logging.info(f"[NextDiT] Context statistics: mean={ctx_mean:.4f}, std={ctx_std:.4f}")
                    
                    # Calculate tokens for NextDiT architecture
                    num_tokens = context.shape[1]  # Use direct sequence length
                    logging.info(f"[NextDiT] Sequence information:")
                    logging.info(f"[NextDiT] - Number of tokens: {num_tokens}")
                    logging.info(f"[NextDiT] - Context final shape: {context.shape}")
                    
                    # Filter out transformer_options since NextDiT uses unified attention
                    kwargs.pop('transformer_options', None)
                    
                    if self.unet is None:
                        raise RuntimeError("NextDiTWrapper's UNet is not set!")
                    
                    # Forward through NextDiT UNet
                    try:
                        output = self.unet(x, timesteps, context, num_tokens=num_tokens, **kwargs)
                        logging.info(f"[NextDiT] Output shape: {output.shape}")
                        return output
                    except Exception as e:
                        logging.error(f"[NextDiT] Error in UNet forward pass: {str(e)}")
                        logging.error(f"[NextDiT] Last known tensor shapes:")
                        logging.error(f" - x: {x.shape}")
                        logging.error(f" - timesteps: {timesteps.shape}")
                        logging.error(f" - context: {context.shape}")
                        raise

            # Create and configure NextDiT wrapper
            next_dit = NextDiTWrapper()
            next_dit.set_unet(unet)
            unet = next_dit

            extra_input = {}
            print(f"✅ Detected Lumina-Image 2.0 model with NextDiT architecture (context_dim={context_dim}, seq_len={context_len})")


        if context_dim is not None:
            input_names = ["x", "timesteps", "context"]
            output_names = ["h"]

            dynamic_axes = {
                "x": {0: "batch", 2: "height", 3: "width"},
                "timesteps": {0: "batch"},
                "context": {0: "batch", 1: "num_embeds"} if not is_lumina2 else {0: "batch", 1: "seq_len"},
            }

            transformer_options = model.model_options['transformer_options'].copy()
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
            else:
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
                has_nan = tensor.isnan().any().item()
                has_inf = tensor.isinf().any().item()
                if has_nan or has_inf:
                    logging.warning(f"[TensorRT] Found NaN/Inf in {name} tensor!")
        
        try:
            logging.info("[TensorRT] Starting ONNX export...")
            torch.onnx.export(
                unet,
                inputs,
                output_onnx,
                verbose=True,  # Enable verbose mode for more detailed export info
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
        logger = trt.Logger(trt.Logger.INFO)
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
        config.progress_monitor = TQDMProgressMonitor()

        # Builder configuration: choose between fast-build (shorter compile) and max-perf (longer compile)
        if fast_build:
            try:
                # Smaller workspace reduces tactic exploration and compile time
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
                logging.info("[TensorRT] [fast_build] Set WORKSPACE to 4 GiB")
            except Exception as e:
                logging.debug(f"[TensorRT] [fast_build] set_memory_pool_limit not available: {e}")
            try:
                if hasattr(config, 'builder_optimization_level'):
                    config.builder_optimization_level = 2
                    logging.info("[TensorRT] [fast_build] builder_optimization_level=2")
            except Exception as e:
                logging.debug(f"[TensorRT] [fast_build] builder_optimization_level not available: {e}")
            try:
                # Restrict tactic sources to reduce timing
                if hasattr(trt, 'TacticSource') and hasattr(config, 'set_tactic_sources'):
                    sources = 0
                    for s in ('CUBLAS_LT',):
                        if hasattr(trt.TacticSource, s):
                            sources |= getattr(trt.TacticSource, s)
                    if sources:
                        config.set_tactic_sources(sources)
                        logging.info("[TensorRT] [fast_build] Enabled tactic sources: cuBLASLt only")
            except Exception as e:
                logging.debug(f"[TensorRT] [fast_build] set_tactic_sources not available: {e}")
        else:
            # Prefer more GPU-heavy tactic search by increasing workspace and optimization level
            try:
                # Allow up to 12 GiB workspace on 24 GiB cards for faster tactic profiling
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
                logging.info("[TensorRT] Set WORKSPACE memory pool limit to 12 GiB")
            except Exception as e:
                logging.debug(f"[TensorRT] set_memory_pool_limit not available: {e}")
            try:
                # Higher optimization level increases GPU kernel search; 4 is aggressive but reasonable
                if hasattr(config, 'builder_optimization_level'):
                    config.builder_optimization_level = 4
                    logging.info("[TensorRT] Set builder_optimization_level=4")
            except Exception as e:
                logging.debug(f"[TensorRT] builder_optimization_level not available: {e}")
            # Enable more tactic sources so TRT can use GPU-optimized kernels broadly
            try:
                if hasattr(trt, 'TacticSource') and hasattr(config, 'set_tactic_sources'):
                    sources = 0
                    for s in ('CUBLAS', 'CUBLAS_LT', 'CUDNN'):
                        if hasattr(trt.TacticSource, s):
                            sources |= getattr(trt.TacticSource, s)
                    if sources:
                        config.set_tactic_sources(sources)
                        logging.info("[TensorRT] Enabled tactic sources: cuBLAS/cuBLASLt/cuDNN")
            except Exception as e:
                logging.debug(f"[TensorRT] set_tactic_sources not available: {e}")

        logging.info("[TensorRT] Configuring optimization profile")
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
            serialized_engine = builder.build_serialized_network(network, config)
            logging.info("[TensorRT] Successfully built TensorRT engine")
            
            if serialized_engine is None:
                logging.error("[TensorRT] Engine serialization failed!")
                return ()
                
            logging.info(f"[TensorRT] Engine size: {len(serialized_engine)} bytes")

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
