// =============================================================================
// SNIPPET: Kernel Launch, Synchronization, Timing, and GFLOP/s
// MILESTONE: 1 — Step 1: CUDA Fundamentals
// TOPIC: How to actually run and benchmark a CUDA kernel
// =============================================================================
//
// THIS SNIPPET COVERS:
//   1. The <<<>>> syntax that actually launches the kernel on the GPU
//   2. Why you must synchronize before measuring time
//   3. Why you run 10 reps and average (not just time one run)
//   4. How to measure wall-clock time in C++
//   5. The GFLOP/s formula and what it tells you
//   6. What GPU utilization % means for your career
// =============================================================================


    // -------------------------------------------------------------------------
    // WARMUP LAUNCH
    // -------------------------------------------------------------------------

    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
//  ^^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^^^^^^^
//  Function name                    Arguments (same as any function call)
//
//  <<<grid, block>>>
//    This is the CUDA kernel launch syntax — the only thing in C++ that looks
//    like this. The triple angle brackets tell the compiler: "don't call this
//    like a normal function — schedule it to run on the GPU."
//
//    Inside the <<<>>>:
//      grid  — the dim3 we computed: how many blocks to launch (64x64 = 4096)
//      block — the dim3 we defined: how many threads per block (16x16 = 256)
//
//    The GPU receives this as: "launch 4096 blocks, each with 256 threads,
//    all running naive_gemm_kernel simultaneously."
//
//    IMPORTANT: this call RETURNS IMMEDIATELY on the CPU.
//    The GPU starts working in the background. The CPU does not wait.
//    This is called "asynchronous execution."
//
//  WHY a warmup run before timing?
//    The first kernel launch often has extra overhead: the GPU driver loads
//    the compiled kernel binary, warms up caches, initialises state.
//    If you time the first run, you're measuring driver setup, not your kernel.
//    Run it once to "warm up", throw away that time, THEN measure.


    cudaDeviceSynchronize();
//  ^^^^^^^^^^^^^^^^^^^^^^^^
//  Blocks the CPU until the GPU finishes ALL work launched so far.
//  Without this, the CPU would race ahead and start timing before
//  the warmup kernel has even finished — corrupting your benchmark.
//
//  Think of it like: the CPU says "go" to the GPU and then immediately
//  checks its watch. cudaDeviceSynchronize says "wait — don't check the
//  watch until the GPU raises its hand and says it's done."
//
//  You'll see this called in two situations:
//    1. After a warmup, to ensure a clean slate before timing starts
//    2. After the timed loop, to ensure all GPU work is done before t1 is read


    // -------------------------------------------------------------------------
    // TIMED BENCHMARK LOOP
    // -------------------------------------------------------------------------

    int REPS = 10;
//  How many times to run the kernel for timing purposes.
//  WHY not just run it once?
//    A single run has noise: OS interrupts, memory bus contention, thermal
//    throttling, background processes. One unlucky run could be 20% slower
//    than normal. Running 10 times and averaging smooths this out.
//    For a proper benchmark you'd use 100+ reps, but 10 is fine for learning.

    auto t0 = std::chrono::high_resolution_clock::now();
//  ^^^^
//  auto
//    C++ keyword: "figure out the type yourself, I don't want to write it."
//    The actual type of `now()` is something like
//    std::chrono::time_point<std::chrono::high_resolution_clock> — too verbose
//    to write manually. `auto` is not lazy; it's practical.
//
//  std::chrono
//    The C++ standard library namespace for time utilities.
//    `std::` means "this lives in the standard library namespace."
//    `chrono::` means "inside the chrono sub-namespace."
//
//  high_resolution_clock
//    The most precise clock available on your system. On Linux/macOS this
//    typically has nanosecond resolution. There's also steady_clock and
//    system_clock — high_resolution_clock is the right choice for benchmarking.
//
//  ::now()
//    The `::` is the "scope resolution operator" — it means "call the `now`
//    function that belongs to high_resolution_clock."
//    Returns a snapshot of the current time as a time_point object.
//    This is your START stopwatch click.

    for (int r = 0; r < REPS; r++) {
        naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
//  Launch the kernel 10 times back to back.
//  Each launch is asynchronous — the CPU fires them all off very quickly.
//  The GPU queues them and runs them one after another.
//  The CPU does NOT wait between launches.

    cudaDeviceSynchronize();
//  NOW wait. The CPU blocks here until the GPU finishes all 10 kernel runs.
//  Only after this returns is it safe to read t1 — otherwise you'd be
//  timing "how fast the CPU fires off 10 async launches" (microseconds)
//  instead of "how long 10 kernel executions take" (the real answer).

    auto t1 = std::chrono::high_resolution_clock::now();
//  Your END stopwatch click. Captured after GPU is confirmed done.


    // -------------------------------------------------------------------------
    // TIME CALCULATION
    // -------------------------------------------------------------------------

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//  t1 - t0
//    Subtracting two time_points gives a `duration` object — the elapsed time.
//    You can't just do t1 - t0 and get a number; you get a duration object
//    that needs to be converted to a unit you want.
//
//  std::chrono::duration<double, std::milli>(...)
//    Converts the raw duration into milliseconds stored as a double.
//    The two template parameters:
//      double      — use 64-bit floating point (not integer — we want decimals)
//      std::milli  — the unit is milliseconds (1/1000 of a second)
//    Other options: std::micro (microseconds), std::nano (nanoseconds), etc.
//
//  .count()
//    Extracts the raw number from the duration object.
//    After this, `ms` is just a plain double like 152.3 (meaning 152.3ms total).
//
//  / REPS
//    Divide by 10 to get the average per-run time.
//    Total time for 10 runs / 10 = average time for 1 run.


    // -------------------------------------------------------------------------
    // GFLOP/s CALCULATION
    // -------------------------------------------------------------------------

    double gflops = 2.0 * M * K * N / (ms * 1e6);
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//  FIRST: what is a FLOP?
//    FLOP = Floating Point OPeration. One multiplication or one addition.
//    GFLOP/s = Giga (billion) FLOPs per second = how fast the GPU is working.
//    This is the standard way to measure compute throughput.
//
//  HOW MANY FLOPs does one GEMM take?
//    For each of the M*N output elements in C, the kernel does:
//      - K multiplications (A[row][k] * B[k][col] for each k)
//      - K additions      (accumulating into acc)
//      Total: 2*K FLOPs per output element
//    Total FLOPs for the whole matrix: 2 * M * K * N
//    For M=K=N=1024: 2 * 1024 * 1024 * 1024 ≈ 2.15 billion FLOPs
//
//  2.0 * M * K * N
//    Computes total FLOPs as a floating-point number.
//    The `2.0` (not `2`) forces the whole expression to use floating-point
//    math — without it, `2 * 1024 * 1024 * 1024` might overflow a 32-bit int.
//
//  (ms * 1e6)
//    Converts milliseconds to the right denominator for GFLOP/s.
//    Let's derive this step by step:
//
//      GFLOP/s = FLOPs / seconds / 1,000,000,000
//
//      seconds = ms / 1000
//
//      So: GFLOP/s = FLOPs / (ms / 1000) / 1e9
//                  = FLOPs * 1000 / ms / 1e9
//                  = FLOPs / ms / 1e6
//                  = FLOPs / (ms * 1e6)    ← that's the formula
//
//    1e6 = 1,000,000 (scientific notation: 1 × 10^6)
//
//  RESULT: a number like 3500 means 3500 GFLOP/s = 3.5 TFLOP/s


    // -------------------------------------------------------------------------
    // PRINTING RESULTS
    // -------------------------------------------------------------------------

    printf("Naive GEMM:  %.2f ms  |  %.1f GFLOP/s\n", ms, gflops);
//  printf — C-style formatted print (from stdio.h)
//  %.2f   — print a float with 2 decimal places  (e.g. 12.34)
//  %.1f   — print a float with 1 decimal place   (e.g. 3500.2)
//  \n     — newline character

    printf("RTX 4090 peak FP32: ~82500 GFLOP/s\n");
//  The theoretical maximum the RTX 4090 can do: 82.5 TFLOP/s for FP32.
//  This is the ceiling. No kernel can exceed this number.
//  Hardcoded here as a reference point.

    double utilization = gflops / 82500.0 * 100.0;
    printf("Utilization: %.4f%%\n", gflops / 82500.0 * 100.0);
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//  utilization = (what we achieved / theoretical max) * 100
//  Tells you: "of all the compute the GPU has available, what % are we using?"
//
//  %.4f%% — print with 4 decimal places. %% prints a literal % sign
//           (a single % would be interpreted as a format specifier).
//
//  WHAT TO EXPECT from this naive kernel:
//    Naive GEMM typically achieves 1-3% utilisation on an RTX 4090.
//    That sounds terrible — and it is. But that's the point.
//    This baseline number is what you're trying to improve.
//    A tiled shared-memory GEMM gets to ~60-70%.
//    cuBLAS (NVIDIA's tuned library) gets to ~85-90%.
//    FlowRT's persistent kernel targets beating cuBLAS on the specific
//    pattern of flow matching's repeated steps.


// =============================================================================
// WHY THIS NUMBER MATTERS FOR FlowRT
// =============================================================================
//
// This utilization % is your first Nsight Compute data point.
// The reason naive GEMM is so slow (1-3%) is memory bandwidth, not compute:
//   - Every thread reads from global GPU memory independently
//   - No data reuse — A[row][k] is read once per output element in its row,
//     but 1024 different rows all need the same A[row][k] columns from B
//   - The L2 cache gets thrashed → ~40% hit rate (the number from the spec)
//
// When you build the persistent trajectory kernel:
//   - x_t stays in L2 between steps instead of being evicted to global memory
//   - L2 hit rate goes from ~40% to >85%
//   - That's the entire story of Contribution 1
//
// You can't tell that story without this baseline number. Run it. Write it down.
// =============================================================================
//
// NUMBERS CHEAT SHEET
//   1e3  = 1,000          (kilo)
//   1e6  = 1,000,000      (mega)
//   1e9  = 1,000,000,000  (giga)
//   1e12 = 1,000,000,000,000 (tera)
//   GFLOP/s = billions of float ops per second
//   TFLOP/s = trillions of float ops per second
//   RTX 4090 peak FP32 = ~82.5 TFLOP/s = 82,500 GFLOP/s
// =============================================================================
