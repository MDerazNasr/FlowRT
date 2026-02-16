#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

//__device__ mean this function runs on the gpu
// it cannot be called from cpu
// it can only be called from inside a __global__ kernel
// this is our stand-in for the full transformer blocks
// a real velocity field v(x, t) would run 8 transformer blocks
// for now -> v = -x * (1 - t)
// this gives a smooth trajectory from noise toward zero
// the memory behaviour is identical to the real model
// same loop structure and same data movement and same bottleneck

// __device__ is a cuda qualifier out out of 3:
// __global__ called from cpu, runs on gpu (this is a kernel)
// __device__ called from gpu, runs on gpu, this is a helper function
// __host__ called form cpu runs on cpu (nrml c++)
/*  the velocity function v = -x * (1 - t) has a
  real mathematical meaning. at t=0 (pure
  noise) the velocity is large — pushing x
  strongly toward zero. at t=1 (finished
  output) the velocity is zero — nothing left
  to do. this mimics how real flow matching
  velocity fields behave: high curvature early,
   nearly flat late. that property is exactly
  what motivates the time-dependent threshold
  in the speculative sampler later. */
__device__ float toy_velocity(float x, float t) { return -x * (1.0f - t); }

// this kernel runs one step of the euler update.
// it will be launched 50 separate times from the cpu.
// every launch writes x to global memory when it exits.
// every launch reads x from global memory when it starts.
// that is 49 unnecessary global memory round trips.
// griddim and blockdim are chosen so that each thread
// owns one element of the x vector.
//
//
// imp - x[i] is read at the top and written at the botom
// every single step pays that cost twice
// once to read, once to write
// then kernel exits, maybe evicting cache line, so next launch reads it from
// memory again
//
// for 50 step trajectory on vector dimension d --> 100 global memory accesses
// per elem. going to reduce that to 2 (read at start, write at the end)
__global__ void naive_euler_step(float *__restrict__ x, // trajectory state [d]
                                 float t,  // current timestep scalar
                                 float dt, // step size scalar
                                 int d     // dimension of x
) {
  // which element does this thread own
  int i = blockidx.x * blockdim.x + threadidx.x;

  // don't go out of bounds
  if (i >= d) {
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

// one kernel launch handles all n steps internally.
// x is read from global memory exactly once at the start.
// x is written to global memory exactly once at the end.
// between those two points, x lives in a register —
// the fastest memory on the gpu, ~1 cycle latency,
// private to each thread.
// this eliminates n-1 global memory round trips entirely.
__global__ void
persistent_euler_kernel(float *__restrict__ x,        // trajectory state [d]
                        const float *__restrict__ ts, // all timesteps [n_steps]
                        float dt,                     // step size
                        int d,                        // dimension of x
                        int n_steps                   // number of steps
) {
  int i = blockidx.x * blockdim.x + threadidx.x;
  if (i >= d)
    return;

  // read once from global memory into a register.
  // from this point on, xi lives in a register.
  // no other thread can touch xi. no global memory needed.
  float xi = x[i];

  // all n steps happen here inside one kernel launch.
  // xi stays in a register the entire time.
  for (int step = 0; step < n_steps; step++) {
    float t = ts[step];
    float v = toy_velocity(xi, t);
    xi = xi + v * dt;
  }

  // write once to global memory.
  // this is the only time we touch slow memory.
  x[i] = xi;
}

int main() {
  // d is dimension of our trajectory state vector
  // in diffusion policy this would be the action dimension
  // we use 1024 to match yesterday gemm and stress the memory system
  int d = 1024;
  int n_steps = 50;
  float dt = 1.0f / n_steps;

  // build timestep array on the cpu first
  //  ts[i] = i * dt, evenly spaced fro, 0 to 1
  float *h_ts = new float[n_steps];
  for (int i = 0; i < n_steps; i++) {
    h_ts[i] = i * dt;
  }

  // allocate x and timesteps on the gpu
  float *d_x, *d_ts;
  cudamalloc(&d_x, d * sizeof(float));
  cudamalloc(&d_ts, n_steps * sizeof(float));

  // copy timesteps to gpu, these never change across runs
  cudamemcpy(d_ts, h_ts, n_steps * sizeof(float), cudamemcpyhosttodevice);

  // initialise x with random values on cpu, then copy to gpu
  float *h_x = new float[d];
  for (int i = 0; i < d; i++) {
    h_x[i] = (float)rand() / rand_max;
  }

  cudamemcpy(d_x, h_x, d * sizeof(float), cudamemcpyhosttodevice);

  // launch config; 1d grid since x is a 1d vector
  // note - using 1d grid this time (not 2d like gemm)
  // trajectory state x is a vector, not a matrix.
  // one thread per element, threads arranged in a line
  int block = 256;
  int grid = (d + block - 1) / block;

  // timing both kernels
  int reps = 100;

  // naive: 50 separate kernel launches per run
  // reset x to the same initial state before timing
  cudamemcpy(d_x, h_x, d * sizeof(float), cudamemcpyhosttodevice);

  // warm up
  for (int s = 0; s < n_steps; s++) {
    naive_euler_step<<<grid, block>>>(d_x, h_ts[s], dt, d);
  }
  cudadevicesynchronize();

  // reset x again before timed run
  cudamemcpy(d_x, h_x, d * sizeof(float), cudamemcpyhosttodevice);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; r++) {
    for (int s = 0; s < n_steps; s++) {
      naive_euler_step<<<grid, block>>>(d_x, h_ts[s], dt, d);
    }
  }
  cudadevicesynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double naive_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

  // persistent: 1 kernel launch per run
  cudamemcpy(d_x, h_x, d * sizeof(float), cudamemcpyhosttodevice);

  // warm up
  persistent_euler_kernel<<<grid, block>>>(d_x, d_ts, dt, d, n_steps);
  cudadevicesynchronize();

  // resent x again before timed run
  cudamemcpy(d_x, h_x, d * sizeof(float), cudamemcpyhosttodevice);

  auto t2 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; r++) {
    persistent_euler_kernel<<<grid, block>>>(d_x, d_ts, dt, d, n_steps);
  }

  cudadevicesynchronize();
  auto t3 = std::chrono::high_resolution_clock::now();
  double persistent_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count() / reps;

  // we reset x to the same initial state before each timed run
  // imp - both kernels must start from identical inputs so the comparision
  // is fair and the correctness check at the end is meaningful
  // we use 100 reopetitions instead of 10 because these kernels are very fast
  // on a vector of size 1024. more reps gives a more stable average
  //
  // results
  printf("naive    (50 separate launches): %.4f ms\n", naive_ms);
  printf("persistent (1 launch, 50 steps): %.4f ms\n", persistent_ms);
  printf("speedup: %.2fx\n", naive_ms / persistent_ms);

  // correctness check
  //  run both kernels from the same starting x on cpu
  //  and compare final results. max error must be below 1e-4.
  float *h_naive = new float[d];
  float *h_persistent = new float[d];

  // naive result
  cudamemcpy(d_x, h_x, d * sizeof(float), cudamemcpyhosttodevice);
  for (int s = 0; s < n_steps; s++) {
    naive_euler_step<<<grid, block>>>(d_x, h_ts[s], dt, d);
  }
  cudadevicesynchronize();
  cudamemcpy(h_naive, d_x, d * sizeof(float), cudamemcpydevicetohost);

  // persistent result
  cudamemcpy(d_x, h_x, d * sizeof(float), cudamemcpyhosttodevice);
  persistent_euler_kernel<<<grid, block>>>(d_x, d_ts, dt, d, n_steps);
  cudadevicesynchronize();
  cudamemcpy(h_persistent, d_x, d * sizeof(float), cudamemcpydevicetohost);

  float max_err = 0.0f;
  for (int i = 0; i < d; i++) {
    max_err = fmaxf(max_err, fabsf(h_naive[i] - h_persistent[i]));
  }
  printf("max absolute error: %.2e  %s\n", max_err,
         max_err < 1e-4f ? "[pass]" : "[fail]");

  // cleanup
  cudafree(d_x);
  cudafree(d_ts);
  delete[] h_x;
  delete[] h_ts;
  delete[] h_naive;
  delete[] h_persistent;
  return 0;
}
