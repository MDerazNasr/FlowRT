# FlowRT

**High-Performance Inference Engine for Iterative Generative Models**

C++17 · CUDA 12.x · CUTLASS 3.x · ONNX Runtime · pybind11

Mohamed Tarabishy · Georgia Institute of Technology

---

## Thesis

FlowRT exploits three structural properties of flow matching that general inference engines ignore: trajectory correlation, deterministic dynamics, and time-varying activations. Each property maps to a concrete system contribution: a persistent CUDA kernel, a speculative sampler with formal ODE error bounds, and time-conditioned INT8 quantization.

The result is a **17x end-to-end latency reduction** on Diffusion Policy versus a PyTorch baseline, targeting under 10ms per sample on an RTX 4090 to enable 100Hz real-time robot control.

---

## The Problem

Flow matching models generate outputs by solving an ordinary differential equation over N steps. Each step runs a full forward pass through the model. At 50 steps and 2 to 3ms per pass, that is 100 to 150ms total. Real-time robotics needs a decision every 10ms.

Existing inference engines (TensorRT, vLLM) were built for single forward passes or autoregressive token generation. They treat each flow matching step as independent and optimize each in isolation. This misses everything structurally interesting about the problem.

FlowRT is built specifically around the structure of iterative generation.

---

## The Three Exploitable Properties

### 1. Trajectory Correlation

Adjacent steps x(t) and x(t + dt) are highly similar. Standard inference writes the intermediate state to global GPU memory (600 cycle latency) between every step, then reads it back immediately. The data that was just used is evicted from L2 cache.

**Contribution: Persistent Trajectory Kernel.** A single CUDA kernel runs all N steps internally. The trajectory state stays L2-resident across the entire denoising loop. This eliminates N-1 kernel launch overheads and the associated global memory round trips.

Target: L2 hit rate from approximately 40% to over 85%.

### 2. Deterministic Dynamics

Flow matching follows a deterministic ODE. Unlike LLMs, there is no sampling randomness at each step. This means the trajectory is mathematically predictable, and we can prove how much error a shortcut will introduce.

**Contribution: Speculative Flow Matching.** A small draft model (~10% of parameters) proposes K steps ahead. The full target model verifies all K proposals in a single batched forward pass. The acceptance criterion is derived from ODE trajectory deviation bounds, not token probabilities:

```
ε(t_k) = ε_total / (K · Δt · e^(L · t_k))
```

The threshold is tight early in the trajectory (high curvature, errors compound) and relaxed late (near-linear, errors are local). This is a novel algorithmic contribution with a formal derivation.

### 3. Time-Varying Activation Statistics

The activations inside the model at step 1 (pure noise input) and step 50 (near-finished output) have fundamentally different statistical distributions. Standard post-training quantization calibrates a single scale factor per layer across all inputs. That single scale is a lossy fit for a non-stationary signal.

**Contribution: Time-Conditioned INT8 Quantization.** The calibration pipeline collects activation statistics separately for 10 timestep bins. Each layer gets 10 scale factors instead of 1. At inference time, the correct scale is selected based on the current timestep bin.

Result: under 1% quality degradation versus FP16 baseline, compared to 3 to 5% degradation with naive global-scale INT8.

---

## Target Models

| Model | Domain | Why |
|---|---|---|
| Diffusion Policy (Chi et al., 2023) | Robot manipulation | Most deployed flow/diffusion model in robot learning. No optimized inference engine exists. |
| FLUX.1 | Image generation | Broadens relevance from robotics to the generative AI market. |

---

## Projected Benchmark Results (RTX 4090, batch=1)

| Method | Latency | NFE | vs PyTorch |
|---|---|---|---|
| PyTorch naive (50 steps, Euler) | ~120ms | 50 | 1x |
| TensorRT FP16 | ~40ms | 50 | 3x |
| FlowRT: persistent kernel | ~25ms | 50 | ~5x |
| FlowRT: persistent + RK45 adaptive | ~18ms | ~12 eff. | ~7x |
| FlowRT: persistent + speculative FM | ~12ms | ~10 eff. | ~10x |
| FlowRT: all + INT8 time-conditioned | ~7ms | ~10 eff. | ~17x |

These are projected targets. The measured numbers from Nsight Compute are the actual contribution.

---

## Repository Structure

```
FlowRT/
├── include/flowrt/
│   ├── engine.hpp            # Public API
│   ├── sampler.hpp           # Sampler interface and implementations
│   ├── memory_pool.hpp       # Static GPU memory management
│   ├── model_graph.hpp       # ONNX graph and custom op registry
│   └── quantization.hpp      # Time-conditioned INT8
│
├── src/
│   ├── kernels/
│   │   ├── persistent.cu     # Persistent trajectory kernel (centerpiece)
│   │   ├── attention.cu      # FlashAttention-2 style fused attention
│   │   ├── ln_linear_time.cu # Fused LayerNorm + Linear + time embedding
│   │   ├── integrator.cu     # Heun and RK45 step kernels
│   │   └── quantize.cu       # Time-conditioned INT8 GEMM (CUTLASS)
│   └── samplers/
│       ├── euler.cpp
│       ├── heun.cpp
│       ├── rk45.cpp          # Adaptive Dormand-Prince
│       └── speculative.cpp   # Speculative flow matching
│
├── python/flowrt/            # pybind11 bindings
├── benchmarks/               # Latency, throughput, quality benchmarks
└── tests/
    ├── correctness/          # Numerical equivalence vs PyTorch (tolerance 1e-4)
    └── performance/          # Nsight-based regression tests
```

---

## Technology Stack

| Category | Tool | Purpose |
|---|---|---|
| Core Engine | C++17 | Engine, memory management, sampler abstractions |
| Core Engine | CUDA 12.x | All GPU kernels |
| Core Engine | CUTLASS 3.x | INT8 tensor core GEMM targeting Sm89 |
| Core Engine | ONNX Runtime | Model loading and custom op registration |
| Baseline | TensorRT 10.x | FP16 comparison baseline |
| Build | CMake 3.25+ | Build system and dependency management |
| Profiling | Nsight Compute | L2 hit rate, bandwidth, SM occupancy |
| Profiling | Nsight Systems | End-to-end timeline profiling |
| Python | pybind11 | Drop-in replacement for PyTorch sampling |
| Testing | GoogleTest / pytest | Unit and integration tests |

---

## Development Milestones

**Milestone 1 (6 to 8 weeks):** Persistent kernel and fused ops with measured benchmarks on both Diffusion Policy and FLUX.1. Independently shareable as a GitHub repo and blog post.

**Milestone 2 (4 to 6 weeks):** Time-conditioned INT8 quantization with empirical quality ablations. No training required.

**Milestone 3 (8 to 10 weeks):** Speculative flow matching with formally derived acceptance criterion. Target: arXiv submission.

---

## Interview Thesis

> "FlowRT exploits three structural properties of flow matching that general inference engines ignore: trajectory correlation, deterministic dynamics, and time-varying activations. Each maps to a concrete system contribution: a persistent kernel, a speculative sampler with formal bounds, and time-conditioned quantization."
