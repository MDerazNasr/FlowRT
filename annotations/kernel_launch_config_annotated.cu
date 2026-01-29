// =============================================================================
// SNIPPET: Kernel Launch Configuration
// MILESTONE: 1 — Step 1: CUDA Fundamentals
// TOPIC: dim3, blocks, grids, and ceiling division
// =============================================================================
//
// THE PROBLEM BEING SOLVED
//
// You have a 1024x1024 output matrix C — over 1 million cells to compute.
// You want the GPU to compute all of them in parallel.
// The GPU doesn't work with individual threads directly — it works with BLOCKS
// of threads. You need to tell it:
//   1. How big is each block?   (the `block` variable)
//   2. How many blocks do I need to cover the whole matrix?  (the `grid` variable)
//
// Think of it like tiling a floor:
//   - Each tile is a block (16x16 threads)
//   - The entire floor is the grid
//   - You need to figure out how many tiles to cover the whole floor
// =============================================================================


    dim3 block(16, 16);
//  ^^^^
//  dim3
//    A CUDA built-in struct (short for "3-dimensional"). It holds three
//    integers: x, y, and z. Used specifically for describing thread/block
//    dimensions. z defaults to 1 if you don't provide it.
//
//    You could declare it as three separate ints, but dim3 keeps them together
//    and the CUDA launch syntax (<<<grid, block>>>) requires dim3 objects.
//
//  block
//    Just the variable name. Convention is to call it `block` or `blockDim`.
//
//  (16, 16)
//    block.x = 16   (16 threads in the x / column direction)
//    block.y = 16   (16 threads in the y / row direction)
//    block.z = 1    (default — we're doing 2D work, not 3D)
//
//    Total threads per block: 16 * 16 = 256 threads.
//
//    WHY 16x16 specifically?
//      GPU hardware executes threads in groups of 32 called "warps".
//      256 threads = 8 warps. This is a sweet spot:
//        - Big enough to hide memory latency (while one warp waits for memory,
//          another can run)
//        - Small enough to fit several blocks on one SM (Streaming Multiprocessor)
//          simultaneously, keeping the GPU busy
//      32x32 = 1024 threads is the hardware maximum per block.
//      16x16 is the most common starting point for 2D kernels.


    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//  grid
//    How many blocks to launch in total, arranged in a 2D grid.
//    grid.x = number of blocks in the column direction (covers N columns)
//    grid.y = number of blocks in the row direction    (covers M rows)
//
//  The formula: (N + block.x - 1) / block.x
//
//    This is INTEGER CEILING DIVISION. Let's understand why it's needed.
//
//    NAIVE APPROACH: N / block.x
//      If N = 1024 and block.x = 16:  1024 / 16 = 64 blocks exactly. Fine.
//      If N = 1000 and block.x = 16:  1000 / 16 = 62 (integer division rounds DOWN).
//      62 blocks * 16 threads = 992 threads. Columns 992–999 never get computed!
//      Your output would have 32 missing values. Silent wrong answer.
//
//    THE FIX: ceiling division — always round UP.
//      Math ceiling: ⌈N / block.x⌉
//      In integer arithmetic (no ceiling function available):
//        ⌈a / b⌉  =  (a + b - 1) / b
//
//    WHY does (a + b - 1) / b work?
//      Add (b-1) to a before dividing. This "pushes" any remainder over the
//      next whole number, so integer division rounds up instead of down.
//
//      Example: N=1000, block.x=16
//        (1000 + 16 - 1) / 16
//        (1000 + 15)     / 16
//         1015           / 16
//         63.4375 → truncates to 63    ← correct! 63 blocks * 16 = 1008 ≥ 1000
//
//      Example: N=1024, block.x=16  (exact fit)
//        (1024 + 15) / 16 = 1039 / 16 = 64.9375 → truncates to 64  ← still correct
//
//      Example: N=1, block.x=16  (extreme case)
//        (1 + 15) / 16 = 16 / 16 = 1  ← correct, we need 1 block for 1 element
//
//  block.x and block.y
//    Accessing the x and y fields of the dim3 struct we just defined.
//    block.x = 16, block.y = 16. Better than hardcoding 16 in two places —
//    if you change block size later, the grid calculation updates automatically.
//
//  RESULT for M=N=K=1024, block=16x16:
//    grid.x = (1024 + 15) / 16 = 64   (64 blocks wide)
//    grid.y = (1024 + 15) / 16 = 64   (64 blocks tall)
//    Total blocks: 64 * 64 = 4096 blocks
//    Total threads: 4096 * 256 = 1,048,576 threads — one per output element.


// =============================================================================
// HOW THIS CONNECTS TO THE KERNEL
// =============================================================================
//
// When you launch the kernel with <<<grid, block>>>, the GPU uses these to
// populate the built-in variables every thread reads:
//
//   blockDim.x = 16     (from `block`)
//   blockDim.y = 16
//   gridDim.x  = 64     (from `grid`)
//   gridDim.y  = 64
//   blockIdx.x = 0..63  (which block am I in — x direction)
//   blockIdx.y = 0..63  (which block am I in — y direction)
//   threadIdx.x = 0..15 (which thread within my block — x direction)
//   threadIdx.y = 0..15 (which thread within my block — y direction)
//
// And inside the kernel:
//   int col = blockIdx.x * blockDim.x + threadIdx.x;
//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//
// The ceiling division means some of those threads land outside the matrix
// bounds — which is exactly why the guard `if (row >= M || col >= N) return;`
// exists. The extra threads are launched but immediately do nothing.
// =============================================================================
//
// VISUAL: 1000x1000 matrix with 16x16 blocks
//
//   ┌──────────────────────────────────────┐
//   │ [16x16][16x16][16x16]...[16x16][16x16] ← last block hangs over edge
//   │ [16x16][16x16][16x16]...[16x16][16x16]
//   │  ...
//   │ [16x16]...                [16x16][16x16] ← bottom-right corner block
//   └──────────────────────────────────────┘ ← hangs over BOTH edges
//
//   The bottom-right corner block has threads covering columns 992-1007
//   and rows 992-1007 — but only 992-999 actually exist. The 8x8 = 64
//   out-of-bounds threads hit the guard and return immediately.
// =============================================================================
