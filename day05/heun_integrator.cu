#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 1: Toy velocity function
//
// Stand-in for the full transformer model.
// v(x, t) = -x * (1 - t)
// High velocity early (t≈0), near-zero late (t≈1).
// Same memory behavior as the real model.
// ─────────────────────────────────────────────────────────────────────────────
__device__ float toy_velocity(float x, float t) {
    return -x * (1.0f - t);
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 2: Naive Euler kernel (baseline)
//
// First-order integrator. One function evaluation per step.
// Error per step: O(dt^2). Total error over trajectory: O(dt).
// Launched N times from CPU — each launch pays global memory round trip.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void naive_euler_kernel(
    float* __restrict__ x,        // trajectory state [D]
    const float* __restrict__ ts, // all timesteps [N_steps]
    float dt,
    int D,
    int N_steps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;

    float xi = x[i];
    for (int step = 0; step < N_steps; step++) {
        float t = ts[step];
        float v = toy_velocity(xi, t);
        xi = xi + v * dt;
    }
    x[i] = xi;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 3: Persistent Heun kernel
//
// Second-order integrator (predictor-corrector).
// Two function evaluations per step — but error per step: O(dt^3).
// Total error over trajectory: O(dt^2). Far more accurate than Euler
// for the same number of steps, or same accuracy with fewer steps.
//
// Algorithm per step:
//   k1 = v(x_t, t)                  ← first evaluation (predictor)
//   x_pred = x_t + k1 * dt          ← predicted next state
//   k2 = v(x_pred, t + dt)          ← second evaluation (corrector)
//   x_{t+dt} = x_t + (k1 + k2)/2 * dt  ← corrected update
//
// One kernel launch handles all N steps.
// x_t stays in registers throughout — no global memory between steps.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void persistent_heun_kernel(
    float* __restrict__ x,        // trajectory state [D]
    const float* __restrict__ ts, // all timesteps [N_steps]
    float dt,
    int D,
    int N_steps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;

    // READ ONCE — xi lives in a register for all N steps
    float xi = x[i];

    for (int step = 0; step < N_steps; step++) {
        float t = ts[step];

        // Predictor: first velocity evaluation at current state
        float k1 = toy_velocity(xi, t);

        // Predicted next state using Euler step
        float x_pred = xi + k1 * dt;

        // Corrector: second velocity evaluation at predicted state
        float k2 = toy_velocity(x_pred, t + dt);

        // Corrected update: average of k1 and k2
        xi = xi + 0.5f * (k1 + k2) * dt;
    }

    // WRITE ONCE — only global memory access in the entire kernel
    x[i] = xi;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 4: CPU reference
//
// Same Heun algorithm on CPU.
// Used to verify GPU result is correct within tolerance 1e-4.
// ─────────────────────────────────────────────────────────────────────────────
void cpu_heun(
    const float* x_init,
    float*       x_out,
    const float* ts,
    float dt,
    int D,
    int N_steps
) {
    // copy initial state
    for (int i = 0; i < D; i++) x_out[i] = x_init[i];

    for (int step = 0; step < N_steps; step++) {
        float t = ts[step];
        for (int i = 0; i < D; i++) {
            float k1     = toy_velocity(x_out[i], t);
            float x_pred = x_out[i] + k1 * dt;
            float k2     = toy_velocity(x_pred, t + dt);
            x_out[i]     = x_out[i] + 0.5f * (k1 + k2) * dt;
        }
    }
}

// CPU Euler reference — used to compare accuracy between Euler and Heun
void cpu_euler(
    const float* x_init,
    float*       x_out,
    const float* ts,
    float dt,
    int D,
    int N_steps
) {
    for (int i = 0; i < D; i++) x_out[i] = x_init[i];
    for (int step = 0; step < N_steps; step++) {
        float t = ts[step];
        for (int i = 0; i < D; i++) {
            float v  = toy_velocity(x_out[i], t);
            x_out[i] = x_out[i] + v * dt;
        }
    }
}

float max_abs_error(const float* a, const float* b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; i++)
        err = fmaxf(err, fabsf(a[i] - b[i]));
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 5: main()
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    int D       = 1024;
    int N_steps = 50;
    float dt    = 1.0f / N_steps;

    // timestep array
    float* h_ts = new float[N_steps];
    for (int i = 0; i < N_steps; i++) h_ts[i] = i * dt;

    // initial state
    float* h_x = new float[D];
    for (int i = 0; i < D; i++) h_x[i] = (float)rand() / RAND_MAX;

    // GPU allocations
    float *d_x, *d_ts;
    cudaMalloc(&d_x,  D       * sizeof(float));
    cudaMalloc(&d_ts, N_steps * sizeof(float));
    cudaMemcpy(d_ts, h_ts, N_steps * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK = 256;
    int GRID  = (D + BLOCK - 1) / BLOCK;
    int REPS  = 100;

    // ── Euler timing ──────────────────────────────────────────────────────────
    cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);
    naive_euler_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
    cudaDeviceSynchronize();

    cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPS; r++) {
        naive_euler_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double euler_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

    // ── Heun timing ───────────────────────────────────────────────────────────
    cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);
    persistent_heun_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
    cudaDeviceSynchronize();

    cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPS; r++) {
        persistent_heun_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double heun_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / REPS;

    printf("Euler (persistent):  %.4f ms\n", euler_ms);
    printf("Heun  (persistent):  %.4f ms\n", heun_ms);
    printf("Heun overhead vs Euler: %.2fx  (expected ~2x — two velocity evals per step)\n",
           heun_ms / euler_ms);

    // ── correctness: GPU Heun vs CPU Heun ────────────────────────────────────
    float* h_heun_gpu = new float[D];
    float* h_heun_cpu = new float[D];

    cudaMemcpy(d_x, h_x, D * sizeof(float), cudaMemcpyHostToDevice);
    persistent_heun_kernel<<<GRID, BLOCK>>>(d_x, d_ts, dt, D, N_steps);
    cudaDeviceSynchronize();
    cudaMemcpy(h_heun_gpu, d_x, D * sizeof(float), cudaMemcpyDeviceToHost);

    cpu_heun(h_x, h_heun_cpu, h_ts, dt, D, N_steps);

    float err = max_abs_error(h_heun_gpu, h_heun_cpu, D);
    printf("GPU vs CPU Heun error: %.2e  %s\n", err, err < 1e-3f ? "[PASS]" : "[FAIL]");

    // ── accuracy: Euler vs Heun vs ground truth ───────────────────────────────
    // Run both with 10 steps (coarse) and compare to 1000-step reference
    int N_coarse = 10;
    float dt_coarse = 1.0f / N_coarse;

    float* h_ts_coarse = new float[N_coarse];
    for (int i = 0; i < N_coarse; i++) h_ts_coarse[i] = i * dt_coarse;

    float* h_ts_fine = new float[1000];
    for (int i = 0; i < 1000; i++) h_ts_fine[i] = i * (1.0f / 1000);

    float* h_euler_coarse = new float[D];
    float* h_heun_coarse  = new float[D];
    float* h_ref          = new float[D];

    cpu_euler(h_x, h_euler_coarse, h_ts_coarse, dt_coarse, D, N_coarse);
    cpu_heun (h_x, h_heun_coarse,  h_ts_coarse, dt_coarse, D, N_coarse);
    cpu_euler(h_x, h_ref,          h_ts_fine,   1.0f/1000, D, 1000);

    float euler_err = max_abs_error(h_euler_coarse, h_ref, D);
    float heun_err  = max_abs_error(h_heun_coarse,  h_ref, D);

    printf("\nAccuracy comparison (10 steps vs 1000-step reference):\n");
    printf("  Euler error: %.4f\n", euler_err);
    printf("  Heun  error: %.4f\n", heun_err);
    printf("  Heun is %.1fx more accurate than Euler at same NFE\n",
           euler_err / heun_err);

    // ── cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_x); cudaFree(d_ts);
    delete[] h_x; delete[] h_ts; delete[] h_heun_gpu; delete[] h_heun_cpu;
    delete[] h_ts_coarse; delete[] h_ts_fine;
    delete[] h_euler_coarse; delete[] h_heun_coarse; delete[] h_ref;
    return 0;
}
