#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ratio>
#include <stdio.h>

__global__ void naive_gemm_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int K, int N) {

  /*
   * row and col relate to output matrix C
   * Each  thread says I own elem C[r][c] and I will compute it
   *
   * then we have a guard if.... which exits coz of how threads are launched
   * they are organised in rectangualr blocks of 16x16. but the matrix might not
   * divide evenly into 16x16. For exmaple if N = 1000, the last block covers
   * columns 992 to 1007 w/out guard threads would wirte to memory they dont own
   * and corrupt data
   *
   * analogy - like assigning workers to a grid of cells. you hire workers in
   * geoups of 16. If your grid is 1000 wide, the last groyup has 16 workers but
   * only 8 cells. The guard tells the extra 8 workers to do nothign
   *
   * */
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < K; k++) {
    acc += A[row * K + k] * B[k * N + col];
  }

  C[row * N + col] = acc;
}

/* This function computes same matrix multiply on the cpu using three plain
 nested loops
 *  not trying to make it fast !!!
 * its job is to gie us a known correct answer to compare against
 * This is how we verify corectness. Every kernel will get checked this way
 * through numerical equiabalence against a CPU ref withini a tolerance of 1e-4
 * if the GPU gives a diff
 *
 *
 * The GPU kernel and this CPU function compute the exact same thing. But
 floating point arithmetic on a GPU is not guaranteed to produce bit-identical
 results to a CPU — the order of operations can differ slightly, which changes
 rounding.

  That is why we use a tolerance of 1e-4 instead of checking for exact equality.
 Keep that in mind — it will come up again when we verify every FlowRT kernel
 later.
 * */

void cpu_gemm(const float *A, const float *B, float *C, int M, int K, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

/*
 *THis function takes the CPU result and the GPU resulkt and finds the largest
 difference between any two corresponfing elements.
 1 numbers and itd its belwo 1e-4 then the kernel is corredcr

 fabsf is abs value for floats. we need magnitude of error without negatives
 \
 fmaxg keeps a running max. after the loop, max_err holds the worst disagreement
 across all mxn elems of matrix. if that worst case is under 1e-4 then kernel is
 correct

 max absolutte error against CPU ref is the 'correctness' test for every single
 kernel in this project you will weite this check many more times
 * */
float max_abs_error(const float *ref, const float *got, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    max_err = fmaxf(max_err, fabsf(ref[i] - got[i]));
  }
  return max_err;
}
// Here we allovcate memory
// copy data to GPU
// launch the kernel, time it, verify result
int main() {
  // The three matrix dimensions
  int M = 1024, K = 1024, N = 1024;
  // int big enought to hold any memory
  // sizeof(float) always returns 4 (32 bits)
  size_t bytes_A = M * K * sizeof(float);
  size_t bytes_B = K * N * sizeof(float);
  size_t bytes_C = M * N * sizeof(float);

  // carves out M*K floats of CPU RAM and returns adress
  // h_ anythign is a host
  // d_ anything is a dveice
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C_gpu = new float[M * N];
  float *h_C_cpu = new float[M * N];

  for (int i = 0; i < M * K; i++) {
    // produces a random float between 0.0 and 1.0
    // need float cast to keep as float
    h_A[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < K * N; i++) {
    h_B[i] = (float)rand() / RAND_MAX;
  }

  float *d_A, *d_B, *d_C;
  // allocates bytes in GPU and vram.
  // & means adress of
  // cudaMalloc needs to write the GPU address of d_..
  // so we pass a pointer to the pointer and not the pointer itself
  cudaMalloc(&d_A, bytes_A);
  cudaMalloc(&d_B, bytes_B);
  cudaMalloc(&d_C, bytes_C);

  // copies data form CPU to GPU over the PCIe bus
  // The direction flag is explicit
  // We only copy A and B, because the kernel will write C from scratch
  cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

  // dim 3 is a CUDA type for 3D dimensions. We only use x and y here since our
  // mateix is 2D block(16...) means each block contains 16x16=256 threada. Each
  // thread owns one element of C.
  /*   The grid calculation is a ceiling division — it figures out how many
  blocks we need to cover the entire matrix. For N=1024 and block.x=16, that is
  exactly 64 blocks in the x direction. But if N were 1000, plain division would
  give 62.5 — you'd miss the last chunk. The formula (N + block.x - 1) / block.x
  rounds up to 63, ensuring full coverage. That is exactly why the guard in the
  kernel exists — those extra threads in the last block have no valid output
  element.*/
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  /*
   *
   * The first launch before the timer is a warm up. The very first kernel
  launch on a GPU pays a one-time JIT compilation cost. If you time that launch
  you get a misleading number. We throw it away.

  cudaDeviceSynchronize() is critical. GPU kernel launches are asynchronous —
  the CPU fires the launch and immediately moves on without waiting. If you
  start the timer, launch the kernel, and stop the timer without synchronizing,
  you are measuring almost nothing. cudaDeviceSynchronize() blocks the CPU until
  the GPU has actually finished.

  The GFLOP/s formula: a 1024×1024×1024 matrix multiply does 2 * M * K * N
  floating point operations — one multiply and one add per element of the inner
  loop. Dividing by time gives throughput
   * */

  naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
  cudaDeviceSynchronize();

  int REPS = 10;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < REPS; r++) {
    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;
  double gflops = 2.0 * M * K * N / (ms * 1e6);
  printf("Naive GEMM: %.2f ms | %.1f GFlop/s\n", ms, gflops);
  printf("RTX 4090 peak FP32: ~82500 GFlop/s\n");
  printf("Utilization: %.4f%%\n", gflops / 82500.0 * 100.0);

  /*
     cudaMemcpy with cudaMemcpyDeviceToHost copies the GPU result back to CPU
  memory so we can compare it against the CPU reference. Every cudaMalloc gets a
  matching cudaFree. Every new gets a matching delete[]. STandaerd practice
   *
   * */
  cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost);
  cpu_gemm(h_A, h_B, h_C_cpu, M, K, N);
  float err = max_abs_error(h_C_cpu, h_C_gpu, M * N);
  printf("Max absolutte error: %.2e %s\n", err,
         err < 1e-3f ? "[PASS]" : "[FAIL]");
}
