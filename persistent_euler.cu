#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

//__device__ mean this function runs on the GPU
// It cannot be called from CPU
// it can only be called from inside a __global__ kernel
// This is our stand-in for the full transformer blocks
// a real velocity field v(x, t) would run 8 transformer blocks
// for now -> v = -x * (1 - t)
// This gives a smooth trajectory from noise toward zero
// The memory behaviour is identical to the real model
// same loop structure and same data movement and same bottleneck

// __device__ is a CUDA qualifier out out of 3:
// __global__ called from CPU, runs on GPU (this is a kernel)
// __device__ called from GPU, runs on GPU, this is a helper function
// __host__ called form CPU runs on CPU (nrml c++)
/*  The velocity function v = -x * (1 - t) has a
  real mathematical meaning. At t=0 (pure
  noise) the velocity is large — pushing x
  strongly toward zero. At t=1 (finished
  output) the velocity is zero — nothing left
  to do. This mimics how real flow matching
  velocity fields behave: high curvature early,
   nearly flat late. That property is exactly
  what motivates the time-dependent threshold
  in the speculative sampler later. */
__device__ float toy_velocity(float x, float t) { return -x * (1.0f - t); }

// This kernel runs ONE step of the Euler update.
// It will be launched 50 separate times from the CPU.
// Every launch writes x to global memory when it exits.
// Every launch reads x from global memory when it starts.
// That is 49 unnecessary global memory round trips.
// gridDim and blockDim are chosen so that each thread
// owns one element of the x vector.
//
//
// IMP - x[i] is read at the top and written at the botom
// every single step pays that cost twice
// once to read, once to write
// then kernel exits, maybe evicting cache line, so next launch reads it from
// memory again
//
// for 50 step trajectory on vector dimension D --> 100 global memory accesses
// per elem. going to reduce that to 2 (read at start, write at the end)
__global__ void naive_euler_step(float *__restrict__ x, // trajectory state [D]
                                 float t,  // current timestep scalar
                                 float dt, // step size scalar
                                 int D     // dimension of x
) {
  // which element does this thread own
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // don't go out of bounds
  if (i >= D) {
    return;
  }

  // read x from global memory (~600 cycle latency)
  float xi = x[i];

  // compute velocity and update
  float v = toy_velocity(xi, t);
  xi = xi + v * dt;

  // write x back to global memory (~600 cycle memory)
  // this data will be read back next launch immedidately
  x[i] = xi;
}

int main() {
  // D is dimension of our trajectory state vector
  // In diffusion policy this would be the action dimension
  // we use 1024 to match yesterday GEMM and stress the memory system
  int D = 1024;
  int N_steps = 50;
  float dt = 1.0f / N_steps;

  // build timestep array on the CPU first
  //  ts[i] = i * dt, evenly spaced fro, 0 to 1
  float *h_ts = new float[N_steps];
  for (int i = 0; i < N_steps; i++) {
    h_ts[i] = i * dt;
  }

  // allocate x and timesteps on the GPU
  float *d_x, *d_ts;
  cudaMalloc(&d_x, D * sizeof(float));
  cudaMalloc(&d_ts, N_steps * sizeof(float));

  // copy timesteps to GPU, these never change across runs
  cudaMemcpyHostToDevice(d_ts, h_ts, N_steps * sizeof(float),
                         cudaMemcpyHostToDevice);

  // initialise x with random values on CPU, then copy to GPU
  float *h_x = new float[D];
  for (int i = 0; i < D; i++) {
    h_x[i] = (float)rand() / RAND_MAX;
  }

  cudaMemcpy(d_x, h_x, D * sizeof(float) * cudaMemcpyHostToDevice);

  // launch config; 1D grid since x is a 1D vector
  // note - using 1D grid this time (not 2d like GEMM)
  // trajectory state x is a vector, not a matrix.
  // One thread per element, threads arranged in a line
  int BLOCK = 256;
  int GRID = (D + BLOCK - 1) / BLOCK;

  // timing both kernels
  int REPS = 100;

  // Naive: 50 separate kernel launches per run
  // Reset x to the same initial state before timing
  cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);

  // warm up
  for (int s = 0; s < N_steps; s++) {
    naive_euler_step<<<GRID, BLOCK>>>(d_x, h_ts[s], dt, D);
  }
  cudaMemcpyHostToDevice();

  // reset x again before timed run
  cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < REPS; r++) {
    for (int s = 0; s < N_steps; s++) {
      naive_euler_step<<<GRID, BLOCK>>>(d_x, h_ts[s], dt, D);
    }
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double naive_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

  // PERSISTENT: 1 kernel launch per run
  cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);

  // warm up
  persistent_euler_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
  cudaDeviceSynchronize();

  // resent x again before timed run
  cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);

  auto t2 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < REPS; r++) {
    persistent_euler_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
  }

  cudaDeviceSynchronize();
  auto t3 = std::chrono::high_resolution_clock::now();
  double persistent_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count() / REPS;

  // we reset x to the same initial state before each timed run
  // imp - both kernels must start from identical inputs so the comparision
  // is fair and the correctness check at the end is meaningful
  // we use 100 reopetitions instead of 10 because these kernels are very fast
  // on a vector of size 1024. More reps gives a more stable average
  //
  //
}
