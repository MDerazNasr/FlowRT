# FlowRT

High-performance inference engine for flow matching models, built in C++17 and CUDA.

Flow matching models (like Diffusion Policy for robotics and FLUX.1 for image generation) generate outputs by running a neural network 50+ times in sequence. General inference engines treat each step independently. FlowRT is built around the structure of iterative generation, exploiting properties that general engines ignore.

## Three Contributions

**Persistent Trajectory Kernel** — a single CUDA kernel runs all N denoising steps internally, keeping intermediate state L2-resident across the full trajectory. Eliminates N-1 kernel launch overheads and global memory round trips. Target: L2 hit rate from ~40% to over 85%.

**Speculative Flow Matching** — a small draft model proposes K steps ahead, the full model verifies all K in one batched pass. Acceptance criterion derived from ODE trajectory deviation bounds with a time-dependent threshold, tighter early in the trajectory where errors compound and looser late where the path is nearly linear.

**Time-Conditioned INT8 Quantization** — activation distributions shift significantly across timesteps. Calibrates 10 scale factors per layer (one per timestep bin) instead of one global scale. Recovers most quality lost by naive INT8 quantization.

## Target

Under 10ms per sample on RTX 4090, enabling 100Hz real-time robot control. Projected 17x latency reduction over a PyTorch baseline on Diffusion Policy.

## Stack

C++17 · CUDA 12.x · CUTLASS 3.x · ONNX Runtime · pybind11 · TensorRT · CMake

## Supported Models

### Diffusion Policy
Flow matching visuomotor policy. Obs+action concatenated input, 8-layer transformer, 50-step Euler denoising loop.

### unifolm-vla DiT Action Head
Flow matching action head from [unifolm-vla](https://github.com/unitreerobotics/unifolm-vla). Cross-attention DiT conditioned on Qwen2.5-VL backbone features. Supports the same VLM + flow matching DiT architecture as GR00T N1.5 and Pi0.

**Export:** `python export_unifolm_dit.py`

The export resolves four blockers that prevent naive ONNX export of the original unifolm-vla codebase:
1. Denoising loop unrolling — loop moved outside the model (`SingleStepActionHead`)
2. `torch.randn` inside inference path — noise is an external input argument
3. `torch.autocast("cuda")` in forward pass — removed; precision set at compile time
4. `BatchFeature` API boundary — plain tensor I/O throughout

The benchmark shows that even with a compiled ONNX model, the Python dispatch loop between steps adds per-step overhead that scales with N. Moving the loop to C++ — the core contribution of the OpenVINO `RobotActionPipeline` — eliminates this on Intel iGPU the same way FlowRT's persistent trajectory kernel eliminates it on NVIDIA.
