// =============================================================================
// FUNCTION: main() — setup section
// MILESTONE: 1 — Step 1: CUDA Fundamentals
// TOPIC: Memory Allocation and Host→Device Transfer
// =============================================================================
//
// THE BIG PICTURE BEFORE ANY CODE
//
// A GPU is a completely separate computer living inside your PC. It has its own
// processor AND its own RAM. Your normal program runs on the CPU and uses CPU
// RAM. The GPU can only touch its own RAM (called "device memory" or VRAM).
//
// This means before the GPU can do any work, you must:
//   1. Allocate memory on the CPU  (new / malloc)
//   2. Fill it with data           (the rand() loops)
//   3. Allocate memory on the GPU  (cudaMalloc)
//   4. Copy data CPU → GPU         (cudaMemcpy HostToDevice)
//   5. Launch the kernel           (the actual computation)
//   6. Copy results GPU → CPU      (cudaMemcpy DeviceToHost)
//   7. Verify and print            (max_abs_error, printf)
//   8. Free all memory             (delete[], cudaFree)
//
// This snippet covers steps 1–4.
//
// NAMING CONVENTION used throughout FlowRT and all CUDA code:
//   h_ prefix = "host"   = lives on the CPU / in CPU RAM
//   d_ prefix = "device" = lives on the GPU / in GPU VRAM
// =============================================================================

int main() {

    // -------------------------------------------------------------------------
    // STEP 1A: Define matrix dimensions
    // -------------------------------------------------------------------------

    int M = 1024, K = 1024, N = 1024;
    //  ^            ^            ^
    //  M = number of rows in A and C
    //  K = number of cols in A = number of rows in B  (the shared inner dim)
    //  N = number of cols in B and C
    //
    //  A is [1024 x 1024]
    //  B is [1024 x 1024]
    //  C is [1024 x 1024]
    //
    //  Total elements in C: 1024 * 1024 = 1,048,576 (~1 million).
    //  This is a realistic size — real transformer weight matrices are often
    //  this shape or larger.


    // -------------------------------------------------------------------------
    // STEP 1B: Compute how many BYTES each matrix needs
    // -------------------------------------------------------------------------

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    // ^^^^^^
    // size_t
    //   A special integer type guaranteed to be big enough to hold any memory
    //   size on your machine. On a 64-bit system it's a 64-bit unsigned integer.
    //   Always use size_t for sizes and byte counts — not int. An int can only
    //   hold up to ~2 billion, but a large model's weights can exceed that easily.
    //
    // sizeof(float)
    //   Returns the number of bytes one float takes — always 4 on any modern
    //   machine (32 bits / 8 bits-per-byte = 4 bytes).
    //   sizeof is a compile-time operator, not a function — it costs nothing
    //   at runtime.
    //
    // M * K * sizeof(float)
    //   1024 * 1024 * 4 = 4,194,304 bytes = 4 MB for matrix A.
    //   Same for B and C.  Total: 12 MB of data to manage.
    //
    // WHY compute bytes separately?
    //   cudaMalloc and cudaMemcpy need exact byte counts, not element counts.
    //   Storing them in named variables makes the code readable and prevents
    //   you from retyping the formula (and maybe getting it wrong) in 3 places.


    // -------------------------------------------------------------------------
    // STEP 2: Allocate memory on the CPU (host)
    // -------------------------------------------------------------------------

    float *h_A     = new float[M * K];
    float *h_B     = new float[K * N];
    float *h_C_gpu = new float[M * N];
    float *h_C_cpu = new float[M * N];
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //
    // float*
    //   Declares a pointer to float. The pointer itself is just an address.
    //   `new float[M * K]` actually carves out M*K floats worth of CPU RAM
    //   and returns the address of the first element. That address is stored
    //   in h_A.
    //
    // new float[M * K]
    //   C++ heap allocation. "Give me enough RAM for M*K floats."
    //   `new` returns a pointer to the first element.
    //   This memory must be manually freed later with `delete[] h_A`.
    //   (Alternatively you could use malloc() from C — same idea, different syntax.)
    //
    // h_A     — input matrix A, on the CPU. Will be filled with random numbers.
    // h_B     — input matrix B, on the CPU. Same.
    // h_C_gpu — where we'll store the GPU's answer after copying it back.
    //           Starts uninitialized — we don't fill it yet.
    // h_C_cpu — where cpu_gemm() will write its answer.
    //           Also starts uninitialized.
    //
    // We need BOTH h_C_gpu and h_C_cpu so we can compare them with max_abs_error.


    // -------------------------------------------------------------------------
    // STEP 3: Fill A and B with random numbers
    // -------------------------------------------------------------------------

    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    //
    // rand()
    //   C standard library function. Returns a random integer between 0 and
    //   RAND_MAX (typically 32767 or 2147483647 depending on platform).
    //
    // RAND_MAX
    //   A constant defined in <stdlib.h> — the largest value rand() can return.
    //
    // (float)rand() / RAND_MAX
    //   This produces a random float between 0.0 and 1.0.
    //   Step by step:
    //     rand()          → some integer, e.g. 18423
    //     (float)18423    → 18423.0f   (the cast converts int to float)
    //     18423.0 / 32767 → ~0.562f
    //   (float) is called a "cast" — you're explicitly telling the compiler
    //   "treat this integer as a float." Without the cast, integer division
    //   would give 0 for every value except RAND_MAX itself.
    //
    // WHY random data?
    //   We don't care what the actual values are — we just need valid inputs to
    //   exercise the kernel. Random [0,1] floats are safe: no overflow, no
    //   special cases, covers a realistic distribution.


    // -------------------------------------------------------------------------
    // STEP 4A: Allocate memory on the GPU (device)
    // -------------------------------------------------------------------------

    float *d_A, *d_B, *d_C;
    // Declares three pointers. At this point they point nowhere valid —
    // they're just uninitialized addresses. cudaMalloc will set them.

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    // ^^^^^^^^^^^^^^^^^^^^^^^
    //
    // cudaMalloc(&pointer, num_bytes)
    //   GPU equivalent of `new`. Allocates num_bytes of memory in GPU VRAM
    //   and writes the address of that memory into the pointer you pass.
    //
    // &d_A
    //   The & means "address of d_A" — we're passing a pointer TO the pointer.
    //   Why? Because cudaMalloc needs to MODIFY d_A (set it to the GPU address).
    //   In C/C++, to modify a variable inside a function you must pass its address.
    //   Think of it like: "here's where I keep my address book — please write the
    //   GPU address into it."
    //
    //   Without the &, you'd pass the value of d_A (garbage at this point),
    //   not its location — cudaMalloc couldn't write back to it.
    //
    // After these three calls:
    //   d_A points to 4 MB of GPU VRAM  (empty / uninitialized)
    //   d_B points to 4 MB of GPU VRAM  (empty / uninitialized)
    //   d_C points to 4 MB of GPU VRAM  (empty — will be written by the kernel)


    // -------------------------------------------------------------------------
    // STEP 4B: Copy data from CPU → GPU
    // -------------------------------------------------------------------------

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //
    // cudaMemcpy(destination, source, num_bytes, direction)
    //   Copies num_bytes of data from source to destination.
    //
    // d_A                    — destination: GPU address (allocated above)
    // h_A                    — source: CPU address (filled with random floats)
    // bytes_A                — how many bytes to copy
    // cudaMemcpyHostToDevice — direction flag. Options are:
    //                            cudaMemcpyHostToDevice   CPU → GPU  (this one)
    //                            cudaMemcpyDeviceToHost   GPU → CPU  (after kernel)
    //                            cudaMemcpyDeviceToDevice GPU → GPU
    //                            cudaMemcpyHostToHost     CPU → CPU
    //
    // WHY don't we copy d_C?
    //   d_C is the output — the kernel will write into it. There's no input
    //   data to copy there. (We do need to zero it out if we wanted guaranteed
    //   clean starting state, but for GEMM the kernel overwrites every element
    //   so it doesn't matter.)
    //
    // This copy travels over the PCIe bus — the physical connection between
    // your CPU and GPU on the motherboard. It's the slowest step in the pipeline.
    // For 4 MB, it takes ~1ms. For large models (gigabytes of weights), this
    // is a real bottleneck — which is why FlowRT keeps tensors GPU-resident
    // after the first load and never copies them back and forth unnecessarily.


// =============================================================================
// WHAT HAPPENS NEXT (not shown in this snippet)
// =============================================================================
//
//   dim3 block(16, 16);
//   dim3 grid((N + 15) / 16, (M + 15) / 16);
//   naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);  // launch!
//
//   cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost);   // pull result
//   cpu_gemm(h_A, h_B, h_C_cpu, M, K, N);                        // reference
//   float err = max_abs_error(h_C_cpu, h_C_gpu, M * N);          // verify
//
//   delete[] h_A; delete[] h_B; delete[] h_C_gpu; delete[] h_C_cpu;
//   cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);                 // cleanup
// =============================================================================
//
// MEMORY LAYOUT SUMMARY
//
//   CPU RAM:                          GPU VRAM:
//   ┌─────────┐                       ┌─────────┐
//   │  h_A    │ ──cudaMemcpy──────▶  │  d_A    │
//   │  h_B    │ ──cudaMemcpy──────▶  │  d_B    │
//   │  h_C_gpu│ ◀─cudaMemcpy──────── │  d_C    │  (after kernel)
//   │  h_C_cpu│ (written by cpu_gemm, stays on CPU)
//   └─────────┘                       └─────────┘
// =============================================================================
