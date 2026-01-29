// =============================================================================
// SNIPPET: Verification, Cleanup, and Return
// MILESTONE: 1 — Step 1: CUDA Fundamentals
// TOPIC: Copying results back, checking correctness, freeing memory
// =============================================================================
//
// WHERE WE ARE IN THE PIPELINE
//
//   ✓ 1. Allocate CPU memory         (new float[])
//   ✓ 2. Fill with random data       (rand() loops)
//   ✓ 3. Allocate GPU memory         (cudaMalloc)
//   ✓ 4. Copy CPU → GPU              (cudaMemcpy HostToDevice)
//   ✓ 5. Launch kernel + benchmark   (<<<grid, block>>>)
//   → 6. Copy GPU → CPU              (this snippet)
//   → 7. Run CPU reference           (this snippet)
//   → 8. Verify correctness          (this snippet)
//   → 9. Free all memory             (this snippet)
//  → 10. Exit                        (this snippet)
// =============================================================================


    // -------------------------------------------------------------------------
    // STEP 6: Copy results back from GPU → CPU
    // -------------------------------------------------------------------------

    cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost);
//             ^^^^^^   ^^^  ^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^
//
//  h_C_gpu
//    Destination: the CPU array we allocated earlier to hold the GPU's answer.
//    It was empty until now — this is the moment it gets filled.
//
//  d_C
//    Source: the GPU array the kernel just wrote its results into.
//
//  bytes_C
//    Number of bytes to copy: M * N * sizeof(float) = 4 MB.
//    Same value we used when allocating and in the HostToDevice copy.
//
//  cudaMemcpyDeviceToHost
//    Direction flag: GPU → CPU. The reverse of what we did before the kernel.
//    After this call, h_C_gpu on the CPU contains the exact same numbers
//    that d_C on the GPU holds.
//
//  NOTE: This call is SYNCHRONOUS — it blocks the CPU until the copy finishes.
//  (Unlike kernel launches which are async.) So no cudaDeviceSynchronize needed here.


    // -------------------------------------------------------------------------
    // STEP 7: Run the CPU reference
    // -------------------------------------------------------------------------

    cpu_gemm(h_A, h_B, h_C_cpu, M, K, N);
//  Plain C++ function call — no GPU involved.
//  Takes the same input matrices (h_A, h_B) and computes the result into h_C_cpu.
//  Slow (three nested loops over 1024³ = 1 billion iterations) but always correct.
//  This is the ground truth we check the GPU result against.
//
//  WHY run this AFTER the GPU kernel, not before?
//    No technical reason — either order works. But conventionally you benchmark
//    first (GPU path) then verify (CPU reference), so the timing code isn't
//    cluttered with correctness code. Also: the CPU reference for 1024x1024 will
//    take several seconds. You don't want that inside your timing loop.


    // -------------------------------------------------------------------------
    // STEP 8: Verify correctness
    // -------------------------------------------------------------------------

    float err = max_abs_error(h_C_cpu, h_C_gpu, M * N);
//  ^^^^^^^^^^
//  Calls the function we annotated earlier.
//  Walks through all M*N elements, finds the biggest |cpu - gpu| difference.
//  Stores it in `err` as a single float.
//
//  M * N = 1024 * 1024 = 1,048,576 elements being compared.

    printf("Max absolute error: %.2e  %s\n", err, err < 1e-3f ? "[PASS]" : "[FAIL]");
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//  printf
//    C-style formatted print. Takes a format string, then the values to insert.
//
//  "Max absolute error: %.2e  %s\n"
//    The format string. Two placeholders:
//      %.2e — scientific notation with 2 decimal places.
//             e.g. 0.0000312 prints as 3.12e-05
//             Used here because errors are tiny numbers like 0.000047 —
//             scientific notation is much more readable than 0.000047
//      %s   — a string (sequence of characters)
//      \n   — newline
//
//  err
//    First value to insert — goes into %.2e slot.
//
//  err < 1e-3f ? "[PASS]" : "[FAIL]"
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    This is the TERNARY OPERATOR — a one-line if/else that produces a value.
//    Structure:  condition ? value_if_true : value_if_false
//
//    err < 1e-3f     — is the error less than 0.001?
//    ?  "[PASS]"     — if yes, use the string "[PASS]"
//    :  "[FAIL]"     — if no,  use the string "[FAIL]"
//
//    The result (either "[PASS]" or "[FAIL]") goes into the %s slot.
//
//    WHY 1e-3 here instead of 1e-4 from earlier comments?
//      1e-3 = 0.001 is a looser tolerance than 1e-4 = 0.0001.
//      For a 1024x1024 GEMM, floating-point errors can accumulate more
//      (you're summing 1024 terms per element). 1e-3 is still very tight
//      and any real bug will produce errors of 1.0 or more — nowhere near
//      the threshold. The exact cutoff doesn't matter much; what matters
//      is that it catches real bugs while ignoring rounding noise.
//
//  EXAMPLE OUTPUT:
//    Max absolute error: 2.34e-05  [PASS]
//    Max absolute error: 4.71e+01  [FAIL]  ← real bug (off by ~47)


    // -------------------------------------------------------------------------
    // STEP 9: Free all memory
    // -------------------------------------------------------------------------

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
//  ^^^^^^^^^
//  Releases GPU VRAM back to the driver.
//  Pair for every cudaMalloc — one cudaFree per allocation.
//  If you skip this in a long-running program, GPU memory fills up and the
//  next cudaMalloc call fails. For a short benchmark it doesn't matter in
//  practice, but it's always correct to clean up.

    delete[] h_A; delete[] h_B; delete[] h_C_gpu; delete[] h_C_cpu;
//  ^^^^^^^^
//  Releases CPU RAM back to the OS.
//  Pair for every `new` — one `delete[]` per allocation.
//  The [] matters: `delete` (no brackets) frees a single object.
//  `delete[]` frees an array. Using the wrong one is undefined behavior.
//
//  ORDER: free GPU memory first, then CPU memory. Not strictly required here,
//  but it's a good habit — GPU resources are usually the scarcer ones.


    // -------------------------------------------------------------------------
    // STEP 10: Return from main
    // -------------------------------------------------------------------------

    return 0;
//  In C++, main() must return an int.
//  0 = success (universal convention on all operating systems).
//  Any non-zero value = failure. If your program crashes or asserts,
//  the shell sees a non-zero exit code and knows something went wrong.
//  Build systems and CI pipelines check this return value automatically.
}


// =============================================================================
// THE COMPLETE PIPELINE — ALL PIECES TOGETHER
// =============================================================================
//
//  CPU                                    GPU
//  ─────────────────────────────────────────────────────────
//  new float[]  →  allocate h_A, h_B     cudaMalloc → allocate d_A, d_B, d_C
//  rand() fill  →  fill h_A, h_B
//  cudaMemcpy   ──────────────────────▶  d_A ← h_A, d_B ← h_B
//  <<<>>>       ──── launch kernel ────▶  kernel runs: writes d_C
//  cudaDeviceSynchronize ←────────────── GPU signals done
//  cudaMemcpy   ◀──────────────────────  h_C_gpu ← d_C
//  cpu_gemm     →  writes h_C_cpu
//  max_abs_error → compare h_C_cpu vs h_C_gpu → print PASS/FAIL
//  cudaFree     →  release d_A, d_B, d_C
//  delete[]     →  release h_A, h_B, h_C_gpu, h_C_cpu
//  return 0
//
// =============================================================================
// WHAT "PASS" MEANS FOR FlowRT
// =============================================================================
//
//  Every kernel you write for this project — persistent trajectory kernel,
//  fused LayerNorm+Linear, Heun integrator — will have a main() or test
//  that follows this exact pattern:
//
//    run GPU kernel → copy back → run CPU reference → max_abs_error → PASS/FAIL
//
//  The spec says: "verify numerical correctness vs PyTorch (tolerance 1e-4)"
//  That's this. CPU reference = PyTorch. GPU result = your kernel.
//  If they match within tolerance: the kernel is correct and you move on.
//  If they don't: there's a bug in your kernel logic, not just rounding noise.
//
//  You just wrote your first complete CUDA program. This is the template
//  for everything that follows.
// =============================================================================
