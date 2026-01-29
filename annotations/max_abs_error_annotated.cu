// =============================================================================
// FUNCTION: max_abs_error
// MILESTONE: 1 — Step 1: CUDA Fundamentals
// TOPIC: Numerical Correctness Verification
// =============================================================================
//
// WHAT DOES THIS DO?
//   Compares two arrays of numbers — a reference (correct) array and a result
//   (GPU output) array — and returns the single largest difference between them.
//
// WHY DO WE NEED IT?
//   After running the GPU kernel, you can't just check if the output is
//   "exactly equal" to the CPU reference. Floating point arithmetic on the GPU
//   can reorder operations slightly vs the CPU, which changes rounding at the
//   last decimal place. The results are mathematically equivalent but the bits
//   may differ slightly. So instead of checking equality, we check: is the
//   biggest difference smaller than our tolerance (1e-4)? If yes — correct.
//
// HOW IT'S USED:
//   float err = max_abs_error(cpu_result, gpu_result, M * N);
//   if (err < 1e-4f) printf("PASS\n");
//   else             printf("FAIL — max error: %f\n", err);
// =============================================================================

float max_abs_error(const float* ref, const float* got, int n) {
// ^^^^^
// float
//   This function returns a single float — the largest error found.
//
// max_abs_error
//   The function's name. Describes exactly what it returns:
//   the MAXimum ABSolute ERROR across all n elements.
//
// const float* ref
//   ref = "reference" — the known-correct answer (your CPU result).
//   float* — a pointer to an array of floats (the array lives elsewhere in
//            memory; ref is just the address of its first element).
//   const  — this function promises not to modify the reference array.
//            It's read-only input.
//
// const float* got
//   got = what the GPU actually produced.
//   Same type as ref — pointer to a read-only float array.
//   The name "got" as opposed to "expected" is a common testing convention:
//   "expected" = what we wanted, "got" = what we actually received.
//
// int n
//   The total number of elements to compare.
//   For a matrix [M x N], you'd pass n = M * N to check every element.

    float max_err = 0.0f;
    // ^^^^^^^^^^^^^^^^^^^^
    // Declares a local variable to track the largest error seen so far.
    // Starts at 0 because we haven't checked anything yet — the worst
    // error we've seen is "no error at all."
    // 0.0f — the `f` suffix means 32-bit float literal (not 64-bit double).

    for (int i = 0; i < n; i++)
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^
    // A plain loop that visits every element, one by one.
    // i starts at 0 (first element) and goes up to n-1 (last element).
    // i < n (not i <= n) because arrays are 0-indexed: last valid index is n-1.

        max_err = fmaxf(max_err, fabsf(ref[i] - got[i]));
        //        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //
        // Let's break this inside-out:
        //
        // ref[i] - got[i]
        //   The difference between the correct answer and the GPU answer
        //   at position i. Could be positive (GPU was too high) or negative
        //   (GPU was too low).
        //
        // fabsf(...)
        //   fabsf = "float absolute value." Strips the sign, so -0.0003
        //   becomes 0.0003. We only care about the SIZE of the error, not
        //   the direction. The `f` at the end means it works on 32-bit floats
        //   specifically (vs fabs which works on doubles).
        //
        // fmaxf(max_err, fabsf(...))
        //   fmaxf = "float max of two values." Keeps whichever is larger:
        //   the current running maximum, or the new error we just computed.
        //   This is how we track "the biggest error seen so far" — each
        //   iteration either keeps the old record or breaks it.
        //   The `f` suffix again = 32-bit float version.
        //
        // max_err = ...
        //   Overwrites max_err with the winner. After the loop finishes,
        //   max_err holds the single worst-case error across all n elements.

    return max_err;
    // Send that worst-case error back to whoever called this function.
    // The caller then checks: if max_err < 1e-4f → PASS, else FAIL.
}


// =============================================================================
// THE FULL PICTURE — WHY 1e-4?
// =============================================================================
//
// 1e-4 means 0.0001 (1 with 4 decimal places of zeros before it).
//
// A 32-bit float has about 7 significant decimal digits of precision.
// The numbers in a GEMM result might be in the range of hundreds or thousands.
// A result of 512.0003 vs 512.0000 is a difference of 0.0003 — well under 1e-4
// and completely acceptable. That tiny gap is just floating-point rounding, not
// a logic error.
//
// If max_abs_error returns something like 14.7, your kernel has a real bug.
// If it returns 0.00003, your kernel is correct.
//
// Every single FlowRT kernel gets checked this way before being trusted.
// =============================================================================
//
// MATH SHORTHAND REFERENCE
//   fabsf(x)     — |x|          (absolute value, float)
//   fmaxf(a, b)  — max(a, b)    (larger of two floats)
//   1e-4f        — 0.0001f      (scientific notation, float)
// =============================================================================
