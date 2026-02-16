#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

// __device__ means this function lives on the GPU.
// It cannot be called from the CPU.
// It is called from inside our fused kernel below.
//
// What it receives:
//   x — the input vector for this row, length D
//   gamma — learned scale parameter, length D
//   beta — learned shift parameter, length D
//   out — where to write the normalized result, length D
//   D  — how many elements are in the vector
//
// What it does:
//   Step 1: compute the mean of all D elements
//   Step 2: compute the variance of all D elements
//   Step 3: normalize each element using mean and variance
//   Step 4: scale and shift using gamma and beta
__device__ void layernorm(
  const float* __restrict__ x, 
  const float* __restrict__ gamma, 
  const float* __restrict__ beta, 
  float* __restrict__ out,
  int D
) {
  // step 1 - compute mean
  // we loop over all of D elems and add them up
  // dividing by D gives the avg val in this vector
  float mean = 0.0f;
  for (int i = 0; i < D; i++) {
    mean += x[i];
  }
  mean /= D;

  //step 2 - compute variance
  //variance measures how spread out the values are
  //for each elemts we compute how far it is from the mean
  //then we square distance, and averge all the squared distances
  //we add 1e-5 (epsilon) to prevent division by 0 later
  float var = 0.0f;
  for (int = 0; i < D; i++) {
    float diff = x[i] - mean;
    var += diff * diff
  }





}
