#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ratio>
#include <stdio.h>

__global__ void naive_gemm_kernel(const float *__restrict__ a,
                                  const float *__restrict__ b,
                                  float *__restrict__ c, int m, int k, int n) {

  /*
   * row and col relate to output matrix c
   * each  thread says i own elem c[r][c] and i will compute it
   *
   * then we have a guard if.... which exits coz of how threads are launched
   * they are organised in rectangualr blocks of 16x16. but the matrix might not
   * divide evenly into 16x16. for exmaple if n = 1000, the last block covers
   * columns 992 to 1007 w/out guard threads would wirte to memory they dont own
   * and corrupt data
   *
   * analogy - like assigning workers to a grid of cells. you hire workers in
   * geoups of 16. if your grid is 1000 wide, the last groyup has 16 workers but
   * only 8 cells. the guard tells the extra 8 workers to do nothign
   *
   * */
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int p = 0; p < k; p++) {
    acc += a[row * k + p] * b[p * n + col];
  }

  c[row * n + col] = acc;
}

/* this function computes same matrix multiply on the cpu using three plain
 nested loops
 *  not trying to make it fast !!!
 * its job is to gie us a known correct answer to compare against
 * this is how we verify corectness. every kernel will get checked this way
 * through numerical equiabalence against a cpu ref withini a tolerance of 1e-4
 * if the gpu gives a diff
 *
 *
 * the gpu kernel and this cpu function compute the exact same thing. but
 floating point arithmetic on a gpu is not guaranteed to produce bit-identical
 results to a cpu — the order of operations can differ slightly, which changes
 rounding.

  that is why we use a tolerance of 1e-4 instead of checking for exact equality.
 keep that in mind — it will come up again when we verify every flowrt kernel
 later.
 * */

void cpu_gemm(const float *a, const float *b, float *c, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float acc = 0.0f;
      for (int p = 0; p < k; p++) {
        acc += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = acc;
    }
  }
}

/*
 *this function takes the cpu result and the gpu resulkt and finds the largest
 difference between any two corresponfing elements.
 1 numbers and itd its belwo 1e-4 then the kernel is corredcr

 fabsf is abs value for floats. we need magnitude of error without negatives
 \
 fmaxg keeps a running max. after the loop, max_err holds the worst disagreement
 across all mxn elems of matrix. if that worst case is under 1e-4 then kernel is
 correct

 max absolutte error against cpu ref is the 'correctness' test for every single
 kernel in this project you will weite this check many more times
 * */
float max_abs_error(const float *ref, const float *got, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    max_err = fmaxf(max_err, fabsf(ref[i] - got[i]));
  }
  return max_err;
}
// here we allovcate memory
// copy data to gpu
// launch the kernel, time it, verify result
int main() {
  // the three matrix dimensions
  int m = 1024, k = 1024, n = 1024;
  // int big enought to hold any memory
  // sizeof(float) always returns 4 (32 bits)
  size_t bytes_a = m * k * sizeof(float);
  size_t bytes_b = k * n * sizeof(float);
  size_t bytes_c = m * n * sizeof(float);

  // carves out m*k floats of cpu ram and returns adress
  // h_ anythign is a host
  // d_ anything is a dveice
  float *h_a = new float[m * k];
  float *h_b = new float[k * n];
  float *h_c_gpu = new float[m * n];
  float *h_c_cpu = new float[m * n];

  for (int i = 0; i < m * k; i++) {
    // produces a random float between 0.0 and 1.0
    // need float cast to keep as float
    h_a[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < k * n; i++) {
    h_b[i] = (float)rand() / RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  // allocates bytes in gpu and vram.
  // & means adress of
  // cudaMalloc needs to write the gpu address of d_..
  // so we pass a pointer to the pointer and not the pointer itself
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // copies data form cpu to gpu over the pcie bus
  // the direction flag is explicit
  // we only copy a and b, because the kernel will write c from scratch
  cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

  // dim 3 is a cuda type for 3d dimensions. we only use x and y here since our
  // mateix is 2d block(16...) means each block contains 16x16=256 threada. each
  // thread owns one element of c.
  /*   the grid calculation is a ceiling division — it figures out how many
  blocks we need to cover the entire matrix. for n=1024 and block.x=16, that is
  exactly 64 blocks in the x direction. but if n were 1000, plain division would
  give 62.5 — you'd miss the last chunk. the formula (n + block.x - 1) / block.x
  rounds up to 63, ensuring full coverage. that is exactly why the guard in the
  kernel exists — those extra threads in the last block have no valid output
  element.*/
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  /*
   *
   * the first launch before the timer is a warm up. the very first kernel
  launch on a gpu pays a one-time jit compilation cost. if you time that launch
  you get a misleading number. we throw it away.

  cudaDeviceSynchronize() is critical. gpu kernel launches are asynchronous —
  the cpu fires the launch and immediately moves on without waiting. if you
  start the timer, launch the kernel, and stop the timer without synchronizing,
  you are measuring almost nothing. cudaDeviceSynchronize() blocks the cpu until
  the gpu has actually finished.

  the gflop/s formula: a 1024×1024×1024 matrix multiply does 2 * m * k * n
  floating point operations — one multiply and one add per element of the inner
  loop. dividing by time gives throughput
   * */

  naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
  cudaDeviceSynchronize();

  int reps = 10;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; r++) {
    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  double gflops = 2.0 * m * k * n / (ms * 1e6);
  printf("naive gemm: %.2f ms | %.1f gflop/s\n", ms, gflops);
  printf("rtx 4090 peak fp32: ~82500 gflop/s\n");
  printf("utilization: %.4f%%\n", gflops / 82500.0 * 100.0);

  /*
     cudaMemcpy with cudaMemcpyDeviceToHost copies the gpu result back to cpu
  memory so we can compare it against the cpu reference. every cudaMalloc gets a
  matching cudafree. every new gets a matching delete[]. standaerd practice
   *
   * */
  cudaMemcpy(h_c_gpu, d_c, bytes_c, cudaMemcpyDeviceToHost);
  cpu_gemm(h_a, h_b, h_c_cpu, m, k, n);
  float err = max_abs_error(h_c_cpu, h_c_gpu, m * n);
  printf("max absolutte error: %.2e %s\n", err,
         err < 1e-3f ? "[pass]" : "[fail]");
}
