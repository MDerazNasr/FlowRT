# =============================================================================
# FUNCTION: profile_inference
# MILESTONE: 1 — Step 2: Identify top 3 bottlenecks
# TOPIC: PyTorch Profiler — seeing exactly where the GPU time goes
# =============================================================================
#
# WHAT THIS DOES
#   Runs the full 50-step inference loop but with a profiler attached.
#   The profiler intercepts every GPU operation, records how long it takes,
#   and gives you a ranked table: "op X took Y ms total, Z% of all GPU time."
#
# WHY THIS MATTERS FOR FlowRT
#   measure_baseline() tells you the TOTAL time: 39.51ms.
#   profile_inference() tells you WHERE that time is: which specific ops.
#   You cannot optimize what you cannot measure. This table is the evidence
#   that justifies which kernels to write. Without it, you're guessing.
#
# REQUIRED IMPORTS (add to top of file):
#   from torch.profiler import profile, record_function, ProfilerActivity
# =============================================================================

from torch.profiler import profile, record_function, ProfilerActivity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# torch.profiler
#   PyTorch's built-in performance measurement tool.
#   Lives in torch.profiler — a sub-module dedicated to profiling.
#   We import three specific things from it:
#
# profile
#   The main context manager. Wraps the code you want to measure.
#   While active, it intercepts every CPU and GPU operation and records
#   its duration. After the block exits, you can query the results.
#
# record_function
#   A labelling tool. Lets you give a name to a block of code so it
#   shows up by that name in the profiler output. Without it, you'd
#   only see low-level op names like "aten::mm" with no context.
#
# ProfilerActivity
#   An enum (a set of named constants) that tells the profiler WHAT to watch:
#     ProfilerActivity.CPU  — measure time spent on the CPU
#     ProfilerActivity.CUDA — measure time spent on the GPU
#   You pass a list of these to the profiler.


def profile_inference(model, obs_dim=20, action_dim=10, n_steps=50):

    device = torch.device("cuda")
    model = model.to(device).eval()

    x = torch.randn(1, obs_dim + action_dim, device=device)
    timesteps = torch.linspace(0, 1, n_steps, device=device)
    # Same setup as measure_baseline — see export_and_baseline_annotated.py


    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       activities=[...]
#         The list of things to profile. We want both:
#           ProfilerActivity.CPU  — how long does the CPU spend launching ops,
#                                   copying data, running Python code?
#           ProfilerActivity.CUDA — how long does the GPU actually spend
#                                   executing each kernel?
#         These two numbers are often very different. A GPU kernel might take
#         5ms of GPU time but be launched in 0.01ms of CPU time. The GPU
#         number is what matters for latency.

        record_shapes=True,
#       ^^^^^^^^^^^^^^^^^^^
#       Tells the profiler to also record the shapes (dimensions) of every
#       tensor involved in each operation. This is crucial for FlowRT:
#       you need to know "this Linear layer was [256 x 256]" to understand
#       why it's slow and what kernel to write for it.
#       Costs a small amount of extra profiling overhead — acceptable.

        with_stack=False
#       ^^^^^^^^^^^^^^^^
#       If True, records the Python call stack for every op — which function
#       called which function called which function. Useful for debugging
#       but very expensive (slows the profiler itself down significantly).
#       False here: we don't need stack traces, just op timings.

    ) as prof:
#   ^^^^^^^^^^
#   `as prof` — binds the profiler object to the name `prof`.
#   After the `with` block exits, `prof` holds all the recorded data.
#   You can then query it with prof.key_averages(), prof.export_chrome_trace(), etc.

        with torch.no_grad():
            for t in timesteps:
                with record_function(f"step"):
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#               record_function("step")
#                 Labels this block as "step" in the profiler output.
#                 Every iteration of the loop gets this label. The profiler
#                 will aggregate all 50 iterations under the name "step"
#                 and show you the total and average time.
#
#               f"step"
#                 An f-string. Here it's just the literal string "step" —
#                 no variables being inserted. You could use f"step_{t:.2f}"
#                 to label each step individually, but that creates 50 separate
#                 entries in the table, which is harder to read.

                    v = model(x, t.unsqueeze(0))
                    x[:, obs_dim:] = x[:, obs_dim:] + v * (1.0 / n_steps)
#               Same forward pass + Euler update as before.
#               The profiler silently records every GPU kernel launched
#               inside these two lines — all 24 of them per step.


    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#   prof.key_averages()
#     Aggregates all recorded events by operation name.
#     If "aten::mm" (matrix multiply) was called 400 times (50 steps × 8 blocks),
#     key_averages() sums them up into one row: "aten::mm: total 18ms, avg 0.045ms"
#     Returns a list of EventAverage objects.
#
#   .table(sort_by="cuda_time_total", row_limit=10)
#     Formats the aggregated data as a readable table.
#
#     sort_by="cuda_time_total"
#       Sort rows by total GPU time (descending). The op that consumed the
#       most GPU time appears first. This immediately tells you your bottleneck.
#       Other sort options: "cpu_time_total", "self_cuda_time_total" (time
#       excluding time spent in called sub-ops).
#
#     row_limit=10
#       Only show the top 10 ops by GPU time. The full table might have 50+
#       entries — most of them tiny. The top 10 account for >95% of the time.
#
#   WHAT THE OUTPUT LOOKS LIKE:
#   -------------------------------------------------------
#   Name            Self CPU   CPU total  Self CUDA  CUDA total  # Calls
#   -------------------------------------------------------
#   aten::mm        0.12ms     0.12ms     12.4ms     12.4ms      400
#   aten::addmm     0.08ms     0.08ms     8.1ms      8.1ms       400
#   aten::layer_norm 0.05ms   0.05ms     5.2ms      5.2ms       400
#   step            2.1ms      38.9ms     0.001ms    38.9ms      50
#   ...
#   -------------------------------------------------------
#   (numbers made up for illustration — run it to see real values)
#
#   HOW TO READ THIS TABLE FOR FlowRT:
#
#   aten::mm / aten::addmm
#     These are matrix multiplications — the nn.Linear layers.
#     If these dominate, the bottleneck is GEMM throughput.
#     This is expected and is what the persistent kernel targets.
#
#   aten::layer_norm
#     This is nn.LayerNorm. If this shows up in the top 3, it confirms
#     that fusing LayerNorm+Linear is worth doing (Milestone 1, Step 4).
#
#   cudaMemcpy / memory operations
#     If data movement shows up, memory bandwidth is the bottleneck.
#     The persistent kernel specifically targets this.
#
#   The top 3 ops by cuda_time_total = the 3 things you build kernels for.
#   The spec is explicit: "identify top 3 bottlenecks" — this table IS that.


# =============================================================================
# BUGS FIXED IN naive_gemm.cu (same session)
# =============================================================================
#
# The following case-sensitivity bugs were all compile errors or silent
# wrong-answer bugs. C/C++ is case-sensitive. CUDA built-ins and macros
# have specific capitalisation that must be exact.
#
#   blockidx    → blockIdx        (CUDA built-in, camelCase)
#   blockdim    → blockDim        (CUDA built-in, camelCase)
#   threadidx   → threadIdx       (CUDA built-in, camelCase)
#   rand_max    → RAND_MAX        (C macro, ALL_CAPS)
#   cudamalloc  → cudaMalloc      (CUDA API, camelCase)
#   cudamemcpy  → cudaMemcpy      (CUDA API, camelCase)
#   cudadevicesynchronize → cudaDeviceSynchronize
#   cudamemcpyhosttodevice → cudaMemcpyHostToDevice
#   cudamemcpydevicetohost → cudaMemcpyDeviceToHost
#
# LOGIC BUG (silent wrong answer — no compile error):
#   for (int k = 0; k < k; k++)
#     The loop variable `k` shadowed the parameter `k`.
#     `k < k` is always false — the loop NEVER ran. acc stayed 0.0f.
#     Every element of C would have been 0. The error check would have
#     caught it (max_abs_error would be ~500), but the code compiled fine.
#   FIXED: renamed loop variable to `p` in both kernel and cpu_gemm:
#     for (int p = 0; p < k; p++)
#     acc += a[row * k + p] * b[p * n + col];
# =============================================================================
