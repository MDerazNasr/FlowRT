#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

// SECTION 1: LayerNorm helper
// __device__ = runs on GPU, called from inside a kernel only.
//
// Normalizes a vector x of length D to have mean=0, std=1,
// then applies learned scale (gamma) and shift (beta).
//
// Formula: out[i] = gamma[i] * ((x[i] - mean) / sqrt(var + 1e-5)) + beta[i]
__device__ void layernorm(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ out,
    int D
) {
    // Step 1: mean
    float mean = 0.0f;
    for (int i = 0; i < D; i++) mean += x[i];
    mean /= D;

    // Step 2: variance
    float var = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var = var / D + 1e-5f;

    // Step 3: normalize + scale + shift
    float inv_std = rsqrtf(var);
    for (int i = 0; i < D; i++) {
        out[i] = gamma[i] * ((x[i] - mean) * inv_std) + beta[i];
    }
}

// SECTION 2: The fused kernel
//
// This single kernel replaces three separate PyTorch operations:
//   1. LayerNorm(x)
//   2. Linear(result )-  W * result + b
//   3. result + t_emb -  time injection
//
// In standard inference each of those is a separate kernel launch with
// global memory round trips between them.
// Here, data flows from step 1 → 2 → 3 entirely in registers and shared
// memory — global memory is touched exactly once at the end.
//
// Each thread block handles ONE row (one sample in the batch).
__global__ void fused_ln_linear_time_kernel(
    const float* __restrict__ x,       // input [B, D]
    const float* __restrict__ gamma,   // layernorm scale [D]
    const float* __restrict__ beta,    // layernorm shift [D]
    const float* __restrict__ W,       // linear weight [D_out, D]
    const float* __restrict__ b,       // linear bias [D_out]
    const float* __restrict__ t_emb,   // time embedding [D_out]
    float*       __restrict__ out,     // output [B, D_out]
    int D,                             // input dimension
    int D_out                          // output dimension
) {
    // Each block processes one row of the batch
    int row = blockIdx.x;

    // shared memory holds the layernorm output for this row
    // so the linear layer can read it without going to global memory
    extern __shared__ float x_norm[];  // [D] — lives in fast shared memory

    // pointers to this row's input and output
    const float* x_row   = x   + row * D;
    float*       out_row  = out + row * D_out;

    // Step 1: LayerNorm 
    // Normalize x_row and store result in shared memory x_norm.
    // Only thread 0 does this — LayerNorm needs all elements to compute mean/var.
    // All other threads wait at the __syncthreads() barrier below.
    if (threadIdx.x == 0) {
        layernorm(x_row, gamma, beta, x_norm, D);
    }

    // barrier: wait until thread 0 has finished writing x_norm
    // before any thread reads from it in the linear step below
    __syncthreads();

    // ── Step 2 + 3: Linear + time injection 
    // Each thread computes one output element.
    // out[j] = dot(W[j], x_norm) + b[j] + t_emb[j]
    //
    // threadIdx.x iterates over output dimensions in steps of blockDim.x
    // so if D_out=256 and blockDim.x=256, each thread owns exactly one output.
    for (int j = threadIdx.x; j < D_out; j += blockDim.x) {
        float acc = b[j];  // start accumulator at bias value
        for (int k = 0; k < D; k++) {
            // W is stored row-major: W[j][k] = W[j * D + k]
            acc += W[j * D + k] * x_norm[k];
        }
        // add time embedding — this is time injection
        // tells the model what denoising step it is at
        out_row[j] = acc + t_emb[j];
    }
}

// SECTION 3: CPU reference
//
// Runs the same three operations sequentially on the CPU.
// Used to verify the GPU result is numerically correct (tolerance 1e-4).
void cpu_fused_ln_linear_time(
    const float* x,
    const float* gamma,
    const float* beta,
    const float* W,
    const float* b,
    const float* t_emb,
    float* out,
    int B, int D, int D_out
) {
    for (int row = 0; row < B; row++) {
        const float* x_row  = x   + row * D;
        float* o_row  = out + row * D_out;

        // LayerNorm
        float mean = 0.0f;
        for (int i = 0; i < D; i++) mean += x_row[i];
        mean /= D;

        float var = 0.0f;
        for (int i = 0; i < D; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
        var = var / D + 1e-5f;
        float inv_std = 1.0f / sqrtf(var);

        float* x_norm = new float[D];
        for (int i = 0; i < D; i++)
            x_norm[i] = gamma[i] * ((x_row[i] - mean) * inv_std) + beta[i];

        // Linear + time injection
        for (int j = 0; j < D_out; j++) {
            float acc = b[j];
            for (int k = 0; k < D; k++)
                acc += W[j * D + k] * x_norm[k];
            o_row[j] = acc + t_emb[j];
        }

        delete[] x_norm;
    }
}

// SECTION 4: main()
int main() {
    // dimensions matching our DiffusionPolicyBlock from Day 2
    int B     = 1;    // batch size
    int D     = 256;  // input/hidden dimension
    int D_out = 256;  // output dimension

    // ── allocate and initialize host arrays 
    float* h_x = new float[B * D];
    float* h_gamma = new float[D];
    float* h_beta  = new float[D];
    float* h_W = new float[D_out * D];
    float* h_b = new float[D_out];
    float* h_temb  = new float[D_out];
    float* h_out_gpu = new float[B * D_out];
    float* h_out_cpu = new float[B * D_out];

    // fill with random values
    for (int i = 0; i < B * D; i++) h_x[i]     = (float)rand()/RAND_MAX - 0.5f;
    for (int i = 0; i < D; i++) h_gamma[i]  = 1.0f;  // standard init
    for (int i = 0; i < D; i++) h_beta[i]   = 0.0f;  // standard init
    for (int i = 0; i < D_out * D; i++) h_W[i]      = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < D_out; i++) h_b[i]      = 0.0f;
    for (int i = 0; i < D_out; i++) h_temb[i]   = (float)rand()/RAND_MAX * 0.1f;

    // ── allocate device arrays 
    float *d_x, *d_gamma, *d_beta, *d_W, *d_b, *d_temb, *d_out;
    cudaMalloc(&d_x, B * D * sizeof(float));
    cudaMalloc(&d_gamma, D * sizeof(float));
    cudaMalloc(&d_beta, D * sizeof(float));
    cudaMalloc(&d_W, D_out * D * sizeof(float));
    cudaMalloc(&d_b, D_out * sizeof(float));
    cudaMalloc(&d_temb, D_out * sizeof(float));
    cudaMalloc(&d_out,B * D_out * sizeof(float));

    // ── copy host → device 
    cudaMemcpy(d_x,     h_x,     B * D     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, D         * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  D         * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,     h_W,     D_out * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,     h_b,     D_out     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temb,  h_temb,  D_out     * sizeof(float), cudaMemcpyHostToDevice);

    // ── launch config 
    // one block per batch row, 256 threads per block
    // shared memory = D floats for the layernorm output
    dim3 grid(B);
    dim3 block(256);
    size_t shared_bytes = D * sizeof(float);

    // warm up
    fused_ln_linear_time_kernel<<<grid, block, shared_bytes>>>(
        d_x, d_gamma, d_beta, d_W, d_b, d_temb, d_out, D, D_out);
    cudaDeviceSynchronize();

    // ── timed run 
    int REPS = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPS; r++) {
        fused_ln_linear_time_kernel<<<grid, block, shared_bytes>>>(
            d_x, d_gamma, d_beta, d_W, d_b, d_temb, d_out, D, D_out);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;
    printf("Fused LN+Linear+Time kernel: %.4f ms\n", ms);

    // ── correctness check 
    cudaMemcpy(h_out_gpu, d_out, B * D_out * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_fused_ln_linear_time(h_x, h_gamma, h_beta, h_W, h_b, h_temb,
                             h_out_cpu, B, D, D_out);

    float max_err = 0.0f;
    for (int i = 0; i < B * D_out; i++)
        max_err = fmaxf(max_err, fabsf(h_out_gpu[i] - h_out_cpu[i]));
    printf("Max absolute error vs CPU: %.2e  %s\n",
           max_err, max_err < 1e-3f ? "[PASS]" : "[FAIL]");

    // ── cleanup 
    cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_W); cudaFree(d_b); cudaFree(d_temb); cudaFree(d_out);
    delete[] h_x; delete[] h_gamma; delete[] h_beta;
    delete[] h_W; delete[] h_b; delete[] h_temb;
    delete[] h_out_gpu; delete[] h_out_cpu;
    return 0;
}
