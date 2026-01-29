// =============================================================================
// FILE: naive_gemm.cu
// MILESTONE: 1 — Step 1: CUDA Fundamentals
// TOPIC: Naive Matrix Multiplication (GEMM) on the GPU
// =============================================================================
//
// WHAT IS GEMM?
//   GEMM = GEneral Matrix Multiply. It computes: C = A * B
//   A is [M x K], B is [K x N], C is [M x N].
//   This is the single most important operation in deep learning —
//   every Linear layer in every neural network is a GEMM.
//   Making this fast is the entire job of inference engineering.
//
// WHY "NAIVE"?
//   This version does the simplest possible thing: one thread per output element,
//   no shared memory tricks, no tiling. It's slow on purpose.
//   Its job is to be CORRECT so we can verify faster versions against it.
// =============================================================================


// --- INCLUDES ----------------------------------------------------------------

#include <chrono>
// ^ C++ standard library for measuring time.
//   You'll use chrono to clock how long the kernel takes.
//   Example: auto t0 = chrono::high_resolution_clock::now();

#include <cuda_runtime.h>
// ^ Gives you everything GPU-related:
//   - Keywords:   __global__, __device__, __shared__
//   - Built-ins:  threadIdx, blockIdx, blockDim, gridDim
//   - Functions:  cudaMalloc, cudaMemcpy, cudaFree, cudaDeviceSynchronize
//   Without this include, the compiler has no idea what any of those words mean.

#include <stdio.h>
// ^ C standard library for printf().
//   Used to print results and error messages to the terminal.


// --- GPU KERNEL --------------------------------------------------------------

__global__ void naive_gemm_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int K, int N) {
// ^^^^^^^^^
// __global__
//   A CUDA keyword (the double underscores are intentional CUDA syntax).
//   Meaning: "this function is CALLED from the CPU but RUNS on the GPU."
//   There are three kinds:
//     __global__  — called from CPU, runs on GPU  ← this one
//     __device__  — called from GPU, runs on GPU
//     __host__    — called from CPU, runs on CPU (normal C++ function)
//
// void
//   This function returns nothing. Results are written into pointer C instead.
//
// const float *__restrict__ A   (same pattern for B and C)
//   float     — each number is a 32-bit floating point value
//   *         — this is a POINTER. A is not the matrix itself — it's a memory
//               address saying "the matrix lives here on the GPU."
//               Think of it like a street address vs the actual house.
//   const     — "I promise not to modify what's at this address."
//               A and B are inputs only. The compiler can optimize more freely.
//   __restrict__ — "I promise A, B, and C point to completely separate memory
//               and never overlap." Lets the compiler skip aliasing safety
//               checks and generate faster instructions. If you lied and they
//               DID overlap, you'd get silent data corruption.
//
// int M, int K, int N
//   The three matrix dimensions:
//     A is [M rows  x  K cols]
//     B is [K rows  x  N cols]
//     C is [M rows  x  N cols]    ← output
//   K is the "inner" dimension that gets summed over in the dot product.

  // --- THREAD INDEX MATH -----------------------------------------------------
  //
  // When you launch this kernel you don't call it once — you launch it
  // THOUSANDS of times simultaneously, one call per thread. Each thread needs
  // to know which output element C[row][col] it is responsible for.
  //
  // The GPU organises threads in a 2-level hierarchy:
  //   Grid  = the entire army (all blocks combined)
  //   Block = a squad (yours is 16x16 = 256 threads per block)
  //   Thread = one soldier inside a block
  //
  // Built-in variables the GPU provides to every thread automatically:
  //   blockIdx.y  — which block am I in? (y = row direction)
  //   blockDim.y  — how many threads per block in the y direction (= 16)
  //   threadIdx.y — which thread within my block am I? (0 to 15)

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  //        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Example with blockDim.y = 16:
  //   Block 0, thread  0  →  row  0
  //   Block 0, thread 15  →  row 15
  //   Block 1, thread  0  →  row 16
  //   Block 1, thread 15  →  row 31
  // Every thread gets a unique row index.

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Same logic, but in the x (column) direction.
  // Every thread now has a unique (row, col) pair → owns one cell of C.


  // --- BOUNDS GUARD ----------------------------------------------------------
  //
  // Threads are launched in rectangular blocks of 16x16. But the matrix might
  // not divide evenly into 16. Example: if N = 1000, the last block covers
  // columns 992 to 1007 — but columns 1000-1007 don't exist. Without this
  // guard, those 8 threads would write to memory they don't own → data
  // corruption or a crash.
  //
  // Analogy: you hire workers in groups of 16. Your grid is 1000 cells wide.
  // The last group has 16 workers but only 8 valid cells. This guard tells
  // the extra 8 workers to do nothing and go home.
  //
  // BUG IN ORIGINAL: `if (row >= M || col >= N): return;`
  //   The colon `:` is Python syntax. C++ does NOT use a colon after if.
  // FIXED VERSION:

  if (row >= M || col >= N) return;
  //                        ^^^^^^ no colon. Just the statement directly.
  // || means OR — if row is out of bounds OR col is out of bounds, exit.


  // --- DOT PRODUCT -----------------------------------------------------------

  float acc = 0.0f;
  // acc = accumulator. Starts at zero, we'll add K terms to it.
  // 0.0f — the `f` suffix means "32-bit float literal" (not 64-bit double).
  //         Always use `f` suffix in CUDA to avoid accidental double arithmetic.

  for (int k = 0; k < K; k++) {
    acc += A[row * K + k] * B[k * N + col];
    //     ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^
    //
    // A[row * K + k]
    //   Matrices are stored in memory as one long flat row of numbers
    //   (row 0 first, then row 1, then row 2, ...). This is called row-major.
    //   To get element A[row][k]: skip (row * K) elements to reach the right
    //   row, then go k more to reach the right column.
    //
    // B[k * N + col]
    //   Same idea. To get B[k][col]: skip (k * N) to reach row k,
    //   then go col more.
    //
    // Together: this is one term of the dot product.
    // We sum K such terms to get the final value of C[row][col].
  }

  C[row * N + col] = acc;
  // Write the final result. ONE write per thread, after all K multiplications.
  // row-major indexing again: skip (row * N) to reach the right row, then col.
}


// --- CPU REFERENCE FUNCTION --------------------------------------------------
//
// Computes the exact same matrix multiply but on the CPU using plain loops.
// NOT trying to be fast — this exists purely to give us a known-correct answer.
//
// Every CUDA kernel in FlowRT gets verified against a CPU reference like this.
// Tolerance is 1e-4 (not exact equality) because floating-point arithmetic on
// the GPU can reorder operations slightly vs the CPU, which changes rounding.
// The math is equivalent but the bits may differ at the 5th decimal place.
//
// BUG IN ORIGINAL: K was missing from the parameter list but used in the loop.
// FIXED VERSION adds `int K`:

void cpu_gemm(const float *A, const float *B, float *C, int M, int K, int N) {
//                                                              ^^^^^
//                                                              was missing

  for (int i = 0; i < M; i++) {       // loop over rows of C
    for (int j = 0; j < N; j++) {     // loop over cols of C
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {   // dot product: sum K terms
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
  // Same row-major indexing as the GPU kernel.
  // O(M * N * K) operations — very slow for large matrices, but always correct.
}


// =============================================================================
// BUGS FOUND AND FIXED IN THIS FILE
// =============================================================================
//
// 1. Kernel guard used Python colon syntax:
//      BEFORE:  if (row >= M || col >= N): return;
//      AFTER:   if (row >= M || col >= N) return;
//
// 2. cpu_gemm missing K in its parameter list:
//      BEFORE:  void cpu_gemm(const float* A, const float* B, float* C, int M, int N)
//      AFTER:   void cpu_gemm(const float* A, const float* B, float* C, int M, int K, int N)
//
// =============================================================================
// WHERE THIS FITS IN FlowRT — MILESTONE 1, STEP 1
// =============================================================================
//
// This naive kernel is the baseline you will profile with Nsight Compute.
// The numbers it produces — L2 hit rate (~40%), global memory transactions,
// SM occupancy — become the "before" numbers. Every optimisation you make
// (tiled shared memory GEMM, persistent trajectory kernel) gets measured
// against this baseline. You can't know if you made something faster unless
// you know where you started.
// =============================================================================
