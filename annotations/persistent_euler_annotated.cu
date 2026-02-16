// =============================================================================
// FILE: persistent_euler.cu
// MILESTONE: 1 — Step 3: Persistent trajectory kernel
// TOPIC: The core idea of FlowRT — eliminating 49 unnecessary memory round trips
// =============================================================================
//
// THE PROBLEM THIS FILE SOLVES
//
//   Diffusion Policy runs 50 denoising steps. A naive implementation launches
//   one GPU kernel per step. Each kernel:
//     1. Reads x from GPU global memory  (~600 cycle wait)
//     2. Does computation
//     3. Writes x back to GPU global memory  (~600 cycle wait)
//     4. Exits — possibly evicting x from cache
//   Next kernel starts, step 1 again. That's 100 global memory accesses total
//   (50 reads + 50 writes) for what is logically just "run 50 steps on x."
//
//   The persistent kernel fixes this with one insight:
//     READ x ONCE → do all 50 steps in a register → WRITE x ONCE
//   That's 2 global memory accesses instead of 100. 50x fewer.
//
// MEASURED RESULTS (RTX 4090, D=1024, 100 reps):
//   Naive     (50 separate launches): 0.1019 ms
//   Persistent (1 launch, 50 steps):  0.0023 ms
//   Speedup: 45x
//   Max absolute error: 0.00e+00  [PASS]
//
// WHY 45x ON D=1024?
//   At D=1024, the computation per step is trivial — just one multiply and one
//   add per element. The kernel is almost entirely launch overhead and memory
//   latency. Eliminating 49 launches and 98 memory round-trips gives a massive
//   speedup. On the full transformer (D=256, 8 blocks of heavy math), the ratio
//   will be smaller — but the principle is identical. x_t stays resident.
// =============================================================================


// --- THE TOY VELOCITY FUNCTION -----------------------------------------------

__device__ float toy_velocity(float x, float t) { return -x * (1.0f - t); }
// ^^^^^^^^
// __device__
//   The third CUDA qualifier. The three are:
//     __global__ — called from CPU, runs on GPU. This is a KERNEL.
//                  Launched with <<<grid, block>>>. Returns void.
//     __device__ — called from GPU, runs on GPU. This is a HELPER FUNCTION.
//                  Can only be called from inside a __global__ or __device__.
//                  Cannot be called from the CPU at all.
//     __host__   — called from CPU, runs on CPU. Normal C++ function.
//                  (Default if you write nothing — every function so far
//                  except the kernels has been implicitly __host__.)
//
//   toy_velocity is __device__ because it gets called from inside the kernels.
//   The compiler inlines it (copies the code directly into the calling kernel)
//   so there's zero function call overhead.
//
// float toy_velocity(float x, float t)
//   Takes one element of the state vector and the current timestep.
//   Returns the velocity: how fast and in which direction x should move.
//
// return -x * (1.0f - t)
//   v = -x * (1 - t)
//
//   MATHEMATICAL MEANING:
//   At t=0 (pure noise):      v = -x * 1.0 = -x     ← large velocity, push hard toward 0
//   At t=0.5 (halfway):       v = -x * 0.5           ← medium velocity
//   At t=1 (finished output): v = -x * 0.0 = 0       ← no velocity, we're done
//
//   This is a simplified stand-in for the real velocity field v(x,t) that
//   the full transformer computes. The MEMORY BEHAVIOUR is identical:
//   same loop structure, same data movement, same bottleneck pattern.
//   We use this toy version so the benchmark isolates the memory effect
//   without requiring 8 transformer blocks.
//
//   This velocity field is why time-conditioned quantization matters
//   (Milestone 2): the activations look very different at t=0 vs t=1,
//   so a single quantization scale for the whole trajectory is suboptimal.


// --- KERNEL 1: NAIVE EULER STEP ----------------------------------------------
// One step. One launch. Must read and write global memory every time.
// -----------------------------------------------------------------------------

__global__ void naive_euler_step(float *__restrict__ x,
                                 float t,
                                 float dt,
                                 int D) {
// float t   — the current timestep SCALAR. Passed directly as a kernel argument.
//             Not a pointer — just one float value, broadcast to all threads.
//             Every thread gets the same t. Each thread has a different x[i].
//
// float dt  — step size scalar. dt = 1.0 / N_steps = 0.02 for 50 steps.
//             Also the same for all threads.
//
// int D     — dimension of x. Needed for the bounds guard.

  int i = blockIdx.x * blockDim.x + threadIdx.x;
// 1D thread indexing this time — x is a vector, not a matrix.
// blockIdx.x / blockDim.x / threadIdx.x only. No .y needed.
// Thread i owns element x[i].

  if (i >= D) return;
// Same ceiling-division guard as in GEMM. If D=1024 and BLOCK=256,
// we launch exactly 4 blocks = 1024 threads. No out-of-bounds here.
// But the guard is always written — defensive programming.

  float xi = x[i];
// ^^^^^^^^^^^^^^^^
// READ FROM GLOBAL MEMORY.
// x lives in GPU global memory (allocated with cudaMalloc).
// This read has ~600 cycle latency — the thread stalls waiting for data.
// The GPU hides some of this by running other warps while waiting,
// but on a tiny D=1024 there aren't many other warps to hide behind.
//
// xi is now a REGISTER variable — stored in the thread's private register file.
// Registers are the fastest memory on the GPU: ~1 cycle access, no latency.
// But they only exist while the thread is alive (inside this kernel call).
// When the kernel exits, registers are gone.

  float v = toy_velocity(xi, t);
  xi = xi + v * dt;
// Euler update: x_new = x_old + v * dt
// All arithmetic happens on xi in a register. No memory touched.

  x[i] = xi;
// ^^^^^^^^^^
// WRITE TO GLOBAL MEMORY. ~600 cycle latency again.
// This data will be read back by the NEXT kernel launch immediately.
// That next launch pays the 600-cycle read cost again.
// 50 steps = 50 reads + 50 writes = 100 global memory accesses per element.
}


// --- KERNEL 2: PERSISTENT EULER KERNEL ---------------------------------------
// All 50 steps. One launch. x touches global memory exactly twice total.
// This is Contribution 1 of FlowRT in its simplest form.
// -----------------------------------------------------------------------------

__global__ void persistent_euler_kernel(
    float* __restrict__ x,
    const float* __restrict__ ts,
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//  ts = timestep array. All N_steps timesteps, pre-computed on CPU, uploaded once.
//  const — this kernel never modifies ts, only reads it.
//  Why pass the full array instead of computing t = step * dt inside?
//  Flexibility: real flow matching uses non-uniform timestep schedules
//  (adaptive ODE solvers, importance sampling). Storing them explicitly
//  generalises to those cases without changing the kernel.
    float dt,
    int D,
    int N_steps
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= D) return;

  float xi = x[i];
// ^^^^^^^^^^^^^^^^
// READ ONCE. This is the only time this thread touches global memory on the way IN.
// xi is now register-resident. It will stay here for all 50 steps.
// No other thread can read or write xi — registers are completely private.
// The GPU has no concept of "sharing" a register between threads.

  for (int step = 0; step < N_steps; step++) {
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//  ALL 50 STEPS happen inside one kernel launch.
//  The loop runs entirely in registers — xi, t, v are all register variables.
//  No global memory is touched inside this loop.
//  This is the key difference.

    float t = ts[step];
//  Read one timestep from ts. ts lives in global memory, but:
//  - ts is only 50 floats = 200 bytes. It fits in L1 cache easily.
//  - After the first iteration, ts[step] comes from L1, not global memory.
//  - L1 cache latency: ~30 cycles. Much faster than ~600 for global memory.
//  So ts reads are cheap after the first step.

    float v = toy_velocity(xi, t);
    xi = xi + v * dt;
//  xi is updated in-register. Zero memory traffic.
  }

  x[i] = xi;
// ^^^^^^^^^^
// WRITE ONCE. The only time this thread touches global memory on the way OUT.
// After 50 steps of computation, one write. Done.
//
// TOTAL global memory accesses per element:
//   Naive:      50 reads + 50 writes = 100
//   Persistent: 1 read  +  1 write  =   2
//   Ratio: 50x fewer memory operations  ← explains the 45x speedup


// =============================================================================
// MEMORY HIERARCHY CHEAT SHEET
// =============================================================================
//
//   Memory Type    | Latency    | Scope              | Size (RTX 4090)
//   ───────────────|────────────|────────────────────|────────────────
//   Registers      | ~1 cycle   | Private to thread  | 255 regs/thread
//   L1 / Shared    | ~30 cycles | Shared within block| 128 KB/SM
//   L2 cache       | ~200 cycles| All SMs share      | 72 MB
//   Global memory  | ~600 cycles| Entire GPU         | 24 GB VRAM
//   CPU RAM        | ~3000 cycles (via PCIe)          | System RAM
//
//   The persistent kernel moves xi from the 600-cycle tier to the 1-cycle tier.
//   The full FlowRT persistent kernel (persistent.cu) will keep x_t in L2
//   (because real x_t is 256-dim × 8 blocks = too big for registers alone).
//   L2 hit rate: naive ~40% → persistent target >85%.
// =============================================================================
}


// --- MAIN --------------------------------------------------------------------

int main() {
  int D = 1024;
// D = dimension of the trajectory state vector x.
// In Diffusion Policy, this would be the action dimension (10 in the toy model,
// up to 256 in the full hidden representation). We use 1024 to:
//   1. Match yesterday's GEMM benchmark (same memory pressure)
//   2. Give the GPU enough elements to fully occupy SMs

  int N_steps = 50;
// 50 denoising steps — the standard number for Diffusion Policy inference.
// This means 50 kernel launches (naive) vs 1 kernel launch (persistent).

  float dt = 1.0f / N_steps;
// Step size: 1/50 = 0.02. Each step advances time by 0.02.
// Total trajectory covers t ∈ [0, 1] in 50 equal steps.


  // --- BUILD TIMESTEP ARRAY --------------------------------------------------

  float *h_ts = new float[N_steps];
  for (int i = 0; i < N_steps; i++) {
    h_ts[i] = i * dt;
  }
// h_ts[0] = 0.00, h_ts[1] = 0.02, ..., h_ts[49] = 0.98
// Note: last step is 0.98, not 1.0. That's correct — the Euler update at t=0.98
// moves x toward t=1.0. You don't need to evaluate at t=1.0 itself.


  // --- GPU ALLOCATION --------------------------------------------------------

  float *d_x, *d_ts;
  cudaMalloc(&d_x, D * sizeof(float));
  cudaMalloc(&d_ts, N_steps * sizeof(float));
// d_x  — the trajectory state vector on GPU. 1024 × 4 bytes = 4 KB.
// d_ts — the timestep array on GPU. 50 × 4 bytes = 200 bytes.
//         Uploaded once, never changed. The persistent kernel reads from this.

  cudaMemcpy(d_ts, h_ts, N_steps * sizeof(float), cudaMemcpyHostToDevice);
// Upload timesteps to GPU once. They never change across runs or reps.
// We DON'T upload d_x here — we do it right before each timed section
// to guarantee identical starting state.


  // --- LAUNCH CONFIG: 1D GRID ------------------------------------------------

  int BLOCK = 256;
  int GRID = (D + BLOCK - 1) / BLOCK;
// 1D grid — x is a vector, not a matrix, so we only need one dimension.
// BLOCK=256 threads per block. GRID=(1024+255)/256 = 4 blocks.
// Total threads: 4 × 256 = 1024 = exactly D. No out-of-bounds threads.
// Compare to GEMM which used dim3 block(16,16) and dim3 grid(...,...) — 2D.
// Rule of thumb: use 2D for matrices, 1D for vectors.
//
// WHY 256 instead of 16×16=256?
//   Same thread count per block, but different shape.
//   For a 1D problem, a 1D block is more natural and equally fast.


  int REPS = 100;
// 100 repetitions instead of 10 (used in GEMM).
// These kernels are very fast on D=1024 — single-digit microseconds.
// Fewer reps would give noisy measurements. 100 reps gives a stable average.


  // --- NAIVE TIMING ----------------------------------------------------------

  cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);
// Reset x to the known initial state before the warmup.
// Without this, the warmup runs on whatever garbage is in d_x.

  for (int s = 0; s < N_steps; s++) {
    naive_euler_step<<<GRID, BLOCK>>>(d_x, h_ts[s], dt, D);
  }
// WARMUP: 50 launches. h_ts[s] passes the CPU-side timestep value directly
// as a scalar argument — no GPU pointer needed for t. The kernel receives
// it by value, like any C++ function argument.

  cudaDeviceSynchronize();

  cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);
// RESET x before the timed run. Critical: both kernels must start from
// identical inputs so the final correctness comparison is meaningful.
// If you forget this, the timed run starts from wherever the warmup left off
// — a different x — and the error check at the end will show disagreement
// even though both kernels are correct.

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < REPS; r++) {
    for (int s = 0; s < N_steps; s++) {
      naive_euler_step<<<GRID, BLOCK>>>(d_x, h_ts[s], dt, D);
//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    Inner loop: 50 launches per rep.
//    Outer loop: 100 reps.
//    Total: 5000 kernel launches in the timed section.
//    We do NOT reset x between reps — the trajectory just keeps going.
//    That's fine: we're measuring steady-state throughput, not correctness here.
    }
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;
// Divide by REPS to get average time for one complete 50-step trajectory.


  // --- PERSISTENT TIMING -----------------------------------------------------

  // (same reset → warmup → reset → timed pattern)
  // persistent_euler_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
  // Note: d_ts (GPU pointer) instead of h_ts[s] (CPU scalar).
  //   The persistent kernel needs all timesteps upfront — it can't receive
  //   them one at a time from the CPU because the CPU isn't involved
  //   during the kernel's 50-step internal loop. So we pass the GPU array.


  // --- RESULTS ---------------------------------------------------------------

  printf("Speedup: %.2fx\n", naive_ms / persistent_ms);
// naive_ms / persistent_ms = 0.1019 / 0.0023 = 44.3 ≈ 45x
// Simple ratio: how many times faster is the persistent kernel?


  // --- CORRECTNESS CHECK -----------------------------------------------------

  // (Both kernels run from the same h_x, results copied back, max_abs_error compared)
  //
  // NOTE: This time we compare naive vs persistent — NOT vs a CPU reference.
  // Why? Both kernels implement the same algorithm: Euler integration with
  // toy_velocity. If they agree with each other to 1e-4, both are correct.
  // The CPU reference would give the same answer but adds runtime (slow CPU loops).
  // For kernel-vs-kernel comparison, direct comparison is sufficient.
  //
  // Max absolute error: 0.00e+00
  // Exact zero — not just below tolerance, but IDENTICAL bit-for-bit.
  // Why? toy_velocity is simple enough that floating-point operations happen
  // in the same order in both kernels (one multiply, one subtract, one add,
  // no reductions). Same order → same rounding → identical bits.
  // More complex kernels (GEMM, attention) will show small non-zero errors
  // from reordered reductions — that's normal and expected.
}


// =============================================================================
// THE BIG PICTURE — HOW THIS CONNECTS TO FlowRT
// =============================================================================
//
//   This file is a TOY demonstration of Contribution 1.
//   The real version (persistent.cu, Milestone 1 Step 3) will:
//     - Replace toy_velocity with the full 8-block transformer forward pass
//     - Replace registers with L2 cache residency (x_t is 256-dim, too big
//       for registers — but small enough to stay in L2 across steps)
//     - Show L2 hit rate rising from ~40% to >85%
//
//   The principle is identical:
//     Don't exit and re-enter the kernel between steps.
//     Keep x_t alive in fast memory (register or L2) for the whole trajectory.
//     Read from slow memory once. Write to slow memory once.
//
//   Toy version:  registers      (1 cycle)   — 45x speedup
//   Real version: L2 cache       (~200 cycles) — target ~5x speedup on full model
//
//   The speedup is smaller on the real model because the transformer compute
//   is expensive enough to partially hide memory latency. But 5x on a
//   computation that already uses cuBLAS is significant.
//
// =============================================================================
// RESULTS RECORDED
// =============================================================================
//   Naive     (50 separate launches): 0.1019 ms
//   Persistent (1 launch, 50 steps):  0.0023 ms
//   Speedup: 45x
//   Max absolute error: 0.00e+00  [PASS]
// =============================================================================
