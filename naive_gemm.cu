#include <chrono>
#include <cuda_runtime.h>
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

  if (row >= M || col >= N): return;

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

void cpu_gemm(const float *A, const float *B, float *C, int M, int N) {
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
 * */
