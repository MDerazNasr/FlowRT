#include <chrono>
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
