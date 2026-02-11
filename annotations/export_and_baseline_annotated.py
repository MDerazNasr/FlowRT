# =============================================================================
# FILE: export_and_baseline.py
# MILESTONE: 1 — Step 2: Export Diffusion Policy to ONNX, measure baseline
# TOPIC: ONNX export, graph validation, latency benchmarking
# =============================================================================
#
# WHAT THIS FILE DOES — TWO JOBS:
#
#   JOB 1: export_to_onnx()
#     Takes the PyTorch model and freezes it into an ONNX file.
#     ONNX = Open Neural Network Exchange — a standard file format that any
#     inference engine (TensorRT, ONNX Runtime, FlowRT) can load and run.
#     Think of it like a PDF for neural networks: the model's structure and
#     weights in a universal format that doesn't require PyTorch to run.
#
#   JOB 2: measure_baseline()
#     Runs the full 50-step denoising loop on GPU, times it 20 times,
#     reports mean/std/min/max latency.
#     This is your "before FlowRT" number. Everything you build gets
#     measured against this.
#
# MEASURED BASELINE (RTX 4090, 50 steps, batch=1, 20 runs):
#   Mean: 39.51 ms  |  Std: 2.52 ms  |  Min: 37.48 ms  |  Max: 46.14 ms
#   Target: <10 ms for 100Hz robot control
#   Gap: 39.51ms / 10ms = ~4x speedup needed just to hit the target.
#   FlowRT's full stack targets ~17x. You have room to spare.
# =============================================================================

import torch
import torch.nn as nn

import onnx
# ^ The `onnx` library lets you inspect and validate ONNX graph files AFTER
#   export. It doesn't run inference — it just reads/writes the file format
#   and checks that the graph is well-formed.
#   Key things it gives you:
#     onnx.load(path)            — reads an .onnx file into memory
#     onnx.checker.check_model() — validates the graph structure
#     model.graph.node/input/output — lets you inspect the graph's contents

import onnxruntime as ort
# ^ ONNX Runtime — the engine that actually RUNS inference from an ONNX file.
#   This is what FlowRT wraps: it loads the ONNX graph, but replaces the
#   bottleneck ops (LayerNorm+Linear+time injection) with our custom CUDA kernels.
#   Think of it as the "player" to onnx's "sheet music file."

import numpy as np
# ^ NumPy — Python's standard library for numerical arrays.
#   Used here just for np.array(latencies) and the .mean()/.std()/.min()/.max()
#   statistics. The latency list from Python is converted to a NumPy array
#   so we can call those stat methods in one line.

import time
# ^ Python standard library for timing.
#   We use time.perf_counter() — the highest resolution timer available in Python.
#   More precise than time.time() (which can have millisecond granularity).
#   perf_counter gives nanosecond resolution on Linux/macOS.


# [TimeEmbedding, DiffusionPolicyBlock, DiffusionPolicyNet are defined here —
#  see diffusion_policy_net_annotated.py for full word-by-word breakdown]


# =============================================================================
# FUNCTION 1: export_to_onnx
# =============================================================================

def export_to_onnx(model, obs_dim=20, action_dim=10, path="diffusion_policy.onnx"):
#                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   path="diffusion_policy.onnx"
#     Default filename for the output file. The caller can override this.
#     .onnx is the standard file extension for ONNX models.

    model.eval()
#   ^^^^^^^^^^^^
#   Switches the model into EVALUATION mode (as opposed to TRAINING mode).
#   Why does this matter? Two layer types behave differently depending on mode:
#     - Dropout: in training, randomly zeros out neurons. In eval: passes through unchanged.
#     - BatchNorm: in training, uses batch statistics. In eval: uses stored running stats.
#   For export and inference, you ALWAYS want eval mode.
#   LayerNorm (which this model uses) is the same in both modes — but call eval()
#   anyway. It's a required habit for any inference code.

    device = torch.device("cuda")
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   torch.device("cuda")
#     Creates a device descriptor for the GPU. The string "cuda" means
#     "use the first available NVIDIA GPU."
#     Could also write "cuda:0" (explicitly GPU #0) or "cpu".
#     Storing it in a variable means you can swap "cuda" → "cpu" in one place
#     if you need to run on a machine without a GPU.

    model = model.to(device)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^
#   Moves all the model's weights (every nn.Linear, nn.LayerNorm parameter)
#   from CPU RAM to GPU VRAM. Same as calling .cuda() on the model.
#   After this line, every computation the model does happens on the GPU.
#   `.to(device)` returns the model itself, so we reassign to `model`.

    x_dummy = torch.randn(1, obs_dim + action_dim, device=device)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   torch.randn(shape..., device=...)
#     Creates a tensor filled with random numbers from a standard normal
#     distribution (mean=0, std=1). Created directly on the GPU.
#   Shape: [1, 30] — batch size 1, observation+action = 20+10 = 30 features.
#   WHY a dummy input? torch.onnx.export traces the model by actually running it
#   with this input and recording every operation. The values don't matter —
#   only the shapes do. This is called "tracing."

    t_dummy = torch.zeros(1, device=device)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   torch.zeros(shape, device=...)
#     Creates a tensor of all zeros. Shape [1] = a single timestep for batch=1.
#   t=0.0 is a valid timestep. Again, the value doesn't matter for tracing.

    torch.onnx.export(
        model,              # the PyTorch model to trace
        (x_dummy, t_dummy), # the dummy inputs — a TUPLE because model.forward(x, t) takes two args
        path,               # where to save the .onnx file
        input_names=["x", "t"],
#       ^^^^^^^^^^^^^^^^^^^^^^^^
#       Names the input nodes in the ONNX graph.
#       Without this, they'd be named "input.1", "input.2" etc.
#       Named inputs make the graph readable and let ONNX Runtime find
#       them by name when you call session.run({"x": ..., "t": ...}).

        output_names=["velocity"],
#       Names the output node. In flow matching, the model predicts the
#       "velocity" — the direction to move in action space. Called velocity
#       because it's the dx/dt in the flow ODE. (Not robot velocity.)

        dynamic_axes={"x": {0: "batch"}, "t": {0: "batch"}, "velocity": {0: "batch"}},
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       By default, ONNX export freezes ALL tensor shapes to exactly what you
#       passed in. That means batch=1 forever — the graph would reject batch=4.
#       dynamic_axes says "dimension 0 of tensor 'x' can vary at runtime, call it 'batch'."
#       {0: "batch"} means: dimension index 0, name it "batch" in the graph.
#       We set this for all three tensors (x, t, velocity).
#       Always do this even when you only use batch=1 — it's good practice
#       and costs nothing.

        opset_version=17,
#       ^^^^^^^^^^^^^^^^^
#       ONNX has versioned sets of ops ("opsets"). Each version adds new ops
#       or changes existing ones. Opset 17 (2022) supports:
#         - LayerNorm as a single op (older opsets decompose it into 5+ ops)
#         - All modern attention patterns
#       Higher = more expressive but requires a newer runtime.
#       17 is the current safe choice for 2024/2025.
    )

    onnx_model = onnx.load(path)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   Reads the .onnx file from disk back into memory as a Python object.
#   This gives us access to the graph structure for inspection and validation.

    onnx.checker.check_model(onnx_model)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   Validates that the ONNX graph is well-formed:
#     - All tensor shapes are consistent
#     - All ops are valid for opset 17
#     - No dangling nodes or missing connections
#   If this passes without raising an exception, the graph is correct.
#   Catches export bugs early — before you waste time loading a broken
#   graph into ONNX Runtime or TensorRT.

    print(f"ONNX export validated: {path}")
    print(f"Graph inputs:{[n.name for n in onnx_model.graph.input]}")
#   onnx_model.graph.input — list of input ValueInfo objects
#   n.name — the name we assigned ("x", "t")
#   [n.name for n in ...] — list comprehension: extract just the names

    print(f"Graph outputs:{[n.name for n in onnx_model.graph.output]}")
    print(f"Graph nodes:{len(onnx_model.graph.node)}")
#   onnx_model.graph.node — list of every primitive operation in the graph.
#   This number will be MUCH larger than 8 (the number of blocks).
#   Each PyTorch operation (LayerNorm, Linear, relu, add) gets decomposed
#   into multiple ONNX primitive ops. Expect ~50-100 nodes.
#   This is NOT a problem — ONNX Runtime fuses many of them back together.


# =============================================================================
# FUNCTION 2: measure_baseline
# =============================================================================

def measure_baseline(model, obs_dim=20, action_dim=10, n_steps=50, n_runs=20):
#   n_steps=50  — 50 denoising steps per inference (the full trajectory)
#   n_runs=20   — repeat the whole 50-step loop 20 times and average

    device = torch.device("cuda")
    model = model.to(device).eval()
#   Chain two calls: move to GPU, then switch to eval mode. Same as above.

    x = torch.randn(1, obs_dim + action_dim, device=device)
    timesteps = torch.linspace(0, 1, n_steps, device=device)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   torch.linspace(start, end, steps, device=...)
#     Creates a 1D tensor of `steps` evenly-spaced values from start to end.
#     linspace(0, 1, 50) → [0.0, 0.0204, 0.0408, ..., 0.9796, 1.0]
#     These are the 50 timesteps of the denoising trajectory.
#     t=0 = pure noise, t=1 = clean action.
#     (Or the reverse depending on convention — direction doesn't matter here.)


    # --- WARMUP --------------------------------------------------------------
    with torch.no_grad():
#   ^^^^^^^^^^^^^^^^^^^^^^
#   torch.no_grad()
#     A context manager. Inside this block, PyTorch does NOT track gradients.
#     During training, PyTorch records every operation so it can compute
#     gradients for backpropagation. This "autograd" tracking uses memory and
#     compute. During inference, you'll never backpropagate, so tracking is
#     pure waste. no_grad() disables it. ALWAYS use this during inference.
#     It's the Python equivalent of the GPU not doing extra bookkeeping work.

        for t in timesteps:
#       Iterates over the 50 timestep values: t=0.0, then t=0.0204, etc.

            v = model(x, t.unsqueeze(0))
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           model(x, t.unsqueeze(0))
#             Calls model.forward(x, t) — one denoising step.
#             Returns v: the predicted velocity [1, action_dim].
#
#           t.unsqueeze(0)
#             t is a scalar tensor (shape []).
#             model.forward expects t of shape [batch] = [1].
#             unsqueeze(0) adds a batch dimension: shape [] → [1].

            x[:, obs_dim:] = x[:, obs_dim:] + v * (1.0 / n_steps)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           This is the EULER INTEGRATOR — the simplest ODE solver.
#           Flow matching defines a trajectory as dx/dt = v(x, t).
#           Euler's method approximates: x_new = x_old + v * Δt
#           where Δt = 1.0 / n_steps (each step is 1/50 of the total time).
#
#           x[:, obs_dim:]
#             Slice notation: "all rows (:), columns from obs_dim onwards."
#             x has shape [1, 30]. Columns 0-19 are the observation (fixed).
#             Columns 20-29 are the noisy action (what we're denoising).
#             We only update the action part — the observation doesn't change.
#
#           Why update x in-place? Because each step's input depends on the
#           previous step's output. This is the "trajectory" — x evolves
#           step by step from noise to clean action.
#
#           This warmup run throws away its timing. Same reason as in CUDA:
#           first run pays JIT compilation cost and cache warmup cost.


    # --- TIMED RUNS ----------------------------------------------------------
    latencies = []
#   Python list to collect one latency measurement per run.

    for _ in range(n_runs):
#   _ is a throwaway variable — we don't need the run index, just repeat 20 times.

        x = torch.randn(1, obs_dim + action_dim, device=device)
#       Fresh random input for each run. Ensures we're not measuring a
#       cached/trivial case. Real inference always starts from fresh noise.

        torch.cuda.synchronize()
#       ^^^^^^^^^^^^^^^^^^^^^^^^
#       Python equivalent of cudaDeviceSynchronize().
#       GPU launches are asynchronous — the CPU fires them and moves on.
#       If you start the timer without synchronizing, you might be measuring
#       "time to queue 50 GPU launches" (microseconds) not "time for the GPU
#       to actually finish all 50 steps" (the real answer).
#       This ensures the GPU has finished any prior work before t0 is recorded.

        t0 = time.perf_counter()
#       ^^^^^^^^^^^^^^^^^^^^^^^^
#       Captures current time in fractional seconds with nanosecond resolution.
#       Use perf_counter() for benchmarking, not time.time().
#       time.time() measures wall-clock time and can jump if the system clock
#       is adjusted. perf_counter() measures a monotonic hardware counter —
#       it never jumps backward and has higher resolution.

        with torch.no_grad():
            for t in timesteps:
                v = model(x, t.unsqueeze(0))
                x[:, obs_dim:] = x[:, obs_dim:] + v * (1.0 / n_steps)
#           The full 50-step denoising loop — same as warmup.

        torch.cuda.synchronize()
#       Blocks CPU until GPU finishes all 50 steps.
#       Without this, t1 is captured while the GPU is still mid-computation.

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
#       t1 - t0 is in seconds. Multiply by 1000 to convert to milliseconds.
#       Append to the list. After 20 runs, latencies = [39.2, 38.9, 40.1, ...]


    latencies = np.array(latencies)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   Converts the Python list to a NumPy array.
#   NumPy arrays have built-in .mean(), .std(), .min(), .max() methods.
#   A Python list doesn't have these — you'd need to import statistics or
#   write your own loops.

    print(f"\nBaseline PyTorch inference (50 steps,Euler)")
    print(f"Mean:{latencies.mean():.2f} ms")
    print(f"Std:{latencies.std():.2f} ms")
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   :.2f — format specifier: float with 2 decimal places.
#   f"..." — f-string: Python's way of embedding expressions in strings.
#            The {} blocks are evaluated at runtime and inserted as strings.
#
#   WHY report std (standard deviation)?
#     High std relative to mean means the GPU is being interrupted —
#     thermal throttling (GPU getting hot and slowing down) or OS processes
#     competing for GPU time. std of 2.52ms on a mean of 39.51ms is ~6%.
#     Acceptable. If std were 10ms+, your measurements would be unreliable.

    print(f"Min:{latencies.min():.2f} ms")
    print(f"Max:{latencies.max():.2f} ms")
    print(f"Target: <10ms for 100Hz robot control")
#   100Hz = 100 decisions per second = 10ms per decision.
#   We're at 39.51ms = 25Hz. The robot would be sluggish and unsafe.


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
#  ^^^^^^^^^^^^^^^^^^^^^^^^
#  Python runs this block ONLY when you execute this file directly:
#    python export_and_baseline.py     ← runs this block
#  NOT when another file imports it:
#    import export_and_baseline        ← skips this block
#  This is the standard Python guard for "runnable script" code.
#  Without it, importing this file would immediately create a model and
#  start benchmarking — unwanted side effects.

    model = DiffusionPolicyNet(
        obs_dim=20,
        action_dim=10,
        hidden_dim=256,
        n_layers=8
    ).cuda()
#   ^^^^^^^^
#   .cuda() is shorthand for .to(torch.device("cuda")).
#   Moves the model's weights to GPU at creation time.
#   Same result as model.to(device) — just more concise.

    print("Exporting to ONNX...")
    export_to_onnx(model)

    print("\nMeasuring baseline latency...")
    measure_baseline(model)


# =============================================================================
# MEASURED BASELINE — YOUR "BEFORE" NUMBER
# =============================================================================
#
#   Baseline PyTorch inference (50 steps, Euler)
#   Mean:  39.51 ms   ← this is your baseline. Write it down.
#   Std:    2.52 ms   ← ~6% variance. GPU is stable, not thermal throttling.
#   Min:   37.48 ms
#   Max:   46.14 ms
#   Target: <10 ms for 100Hz robot control
#
#   GAP ANALYSIS:
#     39.51ms / 10ms = ~4x speedup needed just to hit the MINIMUM target.
#     FlowRT's full stack targets ~17x (from the spec: 120ms→7ms on PyTorch naive).
#     Your PyTorch baseline is already faster than "naive" because PyTorch uses
#     cuBLAS under the hood. So you're starting from a strong baseline —
#     which makes beating it more meaningful.
#
#   WHERE IS THE TIME GOING?
#     50 steps × 8 blocks × 3 ops (LayerNorm + Linear + time_proj)
#     = 1,200 GPU kernel launches per inference call.
#     Each launch has overhead. Each op reads/writes GPU memory separately.
#     FlowRT's fused kernel collapses those 3 ops into 1:
#       1,200 launches → 400 launches  ← that alone is significant
#       3 memory round-trips → 1       ← this is where the real time is saved
#
#   NEXT STEP (Milestone 1, Step 3):
#     Profile with Nsight Compute to confirm:
#       - Which ops take the most time?
#       - What is the memory bandwidth utilization?
#       - What is the L2 cache hit rate?  (expecting ~40% like the GEMM)
#     These numbers become the "before" for the persistent kernel.
# =============================================================================
