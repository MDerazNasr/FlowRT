# =============================================================================
# FILE: diffusion_policy_net.py
# MILESTONE: 1 — Step 2: Export Diffusion Policy to ONNX, identify bottlenecks
# TOPIC: The model you are building FlowRT to accelerate
# =============================================================================
#
# WHAT IS THIS?
#   This is a simplified version of Diffusion Policy (Chi et al., 2023).
#   A robot uses it to decide what action to take next.
#
#   The model takes two inputs:
#     x = [observation, noisy_action] concatenated   (what the robot sees + a noisy guess)
#     t = timestep in [0, 1]                          (how far along the denoising is)
#
#   And outputs:
#     predicted_action — a cleaner version of the noisy action
#
#   This is run 20-100 times per decision (one per denoising step).
#   Each run is one forward pass through this network.
#   FlowRT's goal: make each forward pass as fast as possible.
#
# WHY IT MATTERS FOR FlowRT:
#   Every `nn.Linear` call below is a GEMM under the hood.
#   Every `nn.LayerNorm` + `nn.Linear` + time injection in DiffusionPolicyBlock
#   is EXACTLY the fused kernel you will write in Milestone 1 Step 4
#   (ln_linear_time.cu). You are looking at your target workload right now.
# =============================================================================

import torch
import torch.nn as nn


# =============================================================================
# CLASS 1: TimeEmbedding
# =============================================================================
# PURPOSE: Convert a single timestep number (e.g. 0.7) into a 256-dim vector
#          the network can actually use. A single number carries no structure —
#          you need to project it into a high-dimensional space so the network
#          can distinguish "early denoising step" from "late denoising step."
# =============================================================================

class TimeEmbedding(nn.Module):
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#     class TimeEmbedding
#       Defines a new Python class named TimeEmbedding.
#
#     (nn.Module)
#       TimeEmbedding INHERITS from nn.Module.
#       nn.Module is PyTorch's base class for every neural network component.
#       Inheriting from it gives you:
#         - automatic parameter tracking (PyTorch knows about all your weights)
#         - .to(device) to move weights to GPU
#         - .train() / .eval() mode switching
#         - ability to save/load with torch.save
#       Every network component in PyTorch must inherit from nn.Module.

    def __init__(self, dim):
#       ^^^^^^^^^^^^^^^^^^^
#       __init__ is the constructor — runs once when you create the object.
#       `self` = the object being created (like `this` in C++/Java).
#       `dim` = the size of the embedding vector to produce (e.g. 256).

        super().__init__()
#       ^^^^^^^^^^^^^^^^^^
#       Calls nn.Module's own constructor FIRST, before doing our setup.
#       Required in every PyTorch Module. If you skip this, PyTorch's
#       internal bookkeeping doesn't initialise and things break silently.
#       `super()` = "my parent class" = nn.Module.

        self.linear1 = nn.Linear(1, dim)
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       nn.Linear(in_features, out_features)
#         Creates a fully-connected layer. Internally stores:
#           weight matrix W of shape [dim × 1]
#           bias vector b of shape [dim]
#         When called: output = input @ W.T + b   ← that is a GEMM
#
#       in_features = 1   — input is a single timestep number
#       out_features = dim — output is a dim-dimensional vector
#
#       self.linear1 = ...
#         Attaches it to the object so PyTorch can track the parameters.
#         If you wrote `linear1 = nn.Linear(...)` without `self.`, PyTorch
#         would never see it and the weights wouldn't be saved or trained.

        self.linear2 = nn.Linear(dim, dim)
#       A second linear layer: dim → dim. Adds more expressive power.
#       Two layers with a non-linearity between them can learn more complex
#       mappings than one layer can. (Universal approximation theorem.)

    def forward(self, t):
#       ^^^^^^^^^^^^^^^^^
#       forward() defines what happens when you call the model with input.
#       PyTorch calls this automatically when you do `model(input)`.
#       You never call forward() directly — always use `model(input)`.
#       `t` = the timestep tensor, shape [batch_size] (one timestep per sample).

        t = t.unsqueeze(-1).float()
#           ^^^^^^^^^^^
#           .unsqueeze(-1)
#             Adds a new dimension at position -1 (the last position).
#             Shape goes from [batch_size] → [batch_size, 1]
#             Why? nn.Linear expects input shape [batch, in_features].
#             in_features=1, so we need that trailing dimension of size 1.
#             Without it: shape mismatch error.
#           -1 means "last dimension." Could also write unsqueeze(1) here.
#
#           .float()
#             Ensures the tensor is 32-bit float (FP32).
#             Timesteps might come in as integers or float64 — this normalises.

        t = torch.relu(self.linear1(t))
#           ^^^^^^^^^^^
#           self.linear1(t)
#             Runs the linear layer: t @ W1.T + b1
#             Shape: [batch, 1] → [batch, dim]
#
#           torch.relu(...)
#             ReLU activation: max(0, x). Sets all negative values to 0.
#             Why? Linear layers stacked without activations collapse into one
#             linear layer — the network loses the ability to learn non-linear
#             patterns. ReLU is the non-linearity that allows depth to matter.
#             Shape unchanged: [batch, dim] → [batch, dim]

        return self.linear2(t)
#       Runs the second linear layer: t @ W2.T + b2
#       Shape: [batch, dim] → [batch, dim]
#       No activation after this — the raw embedding is passed as-is to the blocks.


# =============================================================================
# CLASS 2: DiffusionPolicyBlock
# =============================================================================
# PURPOSE: One processing layer. Takes the current signal x and the time
#          embedding t_emb, mixes them, and returns an updated signal.
#          The network stacks 8 of these in sequence.
#
# THIS IS THE HOT PATH. All 8 blocks run every single denoising step.
# FlowRT's fused kernel (ln_linear_time.cu) replaces the three operations
# in forward() with one single CUDA kernel.
# =============================================================================

class DiffusionPolicyBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       LayerNorm normalises the activations before the linear layer.
#       For each sample independently: subtract the mean, divide by std dev,
#       then apply learned scale (gamma) and shift (beta).
#
#       WHY? Without normalisation, activations can explode or vanish as they
#       pass through 8 layers. Values become huge → gradients explode during
#       training. LayerNorm keeps activations in a stable range.
#
#       "Layer" norm (vs batch norm) normalises across the feature dimension
#       for each token/sample independently — works correctly at batch size 1,
#       which matters for robot inference at 100Hz.

        self.linear = nn.Linear(dim, dim)
#       The main transformation: projects the normalised activation.
#       Shape: [batch, dim] → [batch, dim]

        self.time_proj = nn.Linear(dim, dim)
#       Projects the time embedding into the same space as x.
#       This is how time information gets injected into each layer.
#       Shape: [batch, dim] → [batch, dim]

    def forward(self, x, t_emb):
#       x     = current signal, shape [batch, dim]
#       t_emb = time embedding from TimeEmbedding, shape [batch, dim]

        return self.linear(self.norm(x)) + self.time_proj(t_emb)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#       Reading inside-out:
#
#       self.norm(x)
#         Normalise x. Shape: [batch, dim] → [batch, dim]
#
#       self.linear(self.norm(x))
#         Apply the linear transformation to the normalised x.
#         Shape: [batch, dim] → [batch, dim]
#
#       self.time_proj(t_emb)
#         Project the time embedding.
#         Shape: [batch, dim] → [batch, dim]
#
#       ... + ...
#         Element-wise addition. Combines the signal transformation with the
#         time information. This is "time-conditioned" processing — the output
#         of each block depends on BOTH what x is AND what time step we're at.
#
#       WHY ADDITION and not concatenation?
#         Concatenation would double the dimension. Addition keeps it at dim
#         while still mixing both sources of information. This is the standard
#         pattern in diffusion models.
#
#       THE FlowRT CONNECTION:
#         These three operations — norm, linear, time_proj — are currently
#         three separate PyTorch calls. Each one:
#           1. Reads x from GPU memory
#           2. Does its computation
#           3. Writes the result back to GPU memory
#         Then the next operation reads that result back again.
#         That's 6 memory round-trips for what is logically one operation.
#         ln_linear_time.cu fuses all three into one kernel:
#           read x once → normalise → linear → add time → write once
#         This is called "operator fusion" and it's a core FlowRT optimisation.


# =============================================================================
# CLASS 3: DiffusionPolicyNet
# =============================================================================
# PURPOSE: The full network. Wires together input projection, time embedding,
#          8 processing blocks, and output projection.
# =============================================================================

class DiffusionPolicyNet(nn.Module):

    def __init__(self, obs_dim=20, action_dim=10, hidden_dim=256, n_layers=8):
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       obs_dim=20    — the robot's observation vector has 20 numbers
#                       (joint positions, velocities, camera features, etc.)
#       action_dim=10 — the action vector has 10 numbers
#                       (joint torques or target positions for 10 joints)
#       hidden_dim=256 — all internal representations use 256 dimensions
#       n_layers=8    — stack 8 DiffusionPolicyBlocks
#
#       The = signs give DEFAULT values. If you create DiffusionPolicyNet()
#       with no arguments, these values are used automatically.

        super().__init__()

        self.input_proj = nn.Linear(obs_dim + action_dim, hidden_dim)
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       obs_dim + action_dim = 20 + 10 = 30
#       Projects the concatenated [observation, noisy_action] vector from
#       30 dimensions up to 256 (hidden_dim). This is the "embedding" step —
#       lifting the raw inputs into the high-dimensional space where the
#       network does all its processing.

        self.time_emb = TimeEmbedding(hidden_dim)
#       Creates the time embedding module (Class 1 above).
#       Will convert each scalar timestep into a 256-dim vector.

        self.blocks = nn.ModuleList([
            DiffusionPolicyBlock(hidden_dim) for _ in range(n_layers)
        ])
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#       [DiffusionPolicyBlock(hidden_dim) for _ in range(n_layers)]
#         Python list comprehension. Creates a list of 8 DiffusionPolicyBlock
#         objects. `_` is a throwaway loop variable — we don't need the index,
#         just want to repeat 8 times.
#
#       nn.ModuleList([...])
#         Like a Python list, but PyTorch-aware. If you used a plain Python
#         list, PyTorch wouldn't know about the blocks and wouldn't track
#         their parameters (they'd be invisible to the optimiser and wouldn't
#         be saved). nn.ModuleList registers each block properly.

        self.output_proj = nn.Linear(hidden_dim, action_dim)
#       Projects from 256 dimensions back down to 10 (action_dim).
#       This is the final layer — converts the network's internal representation
#       back into an actual action the robot can execute.

    def forward(self, x, t):
#       x = [batch, obs_dim + action_dim] = [batch, 30]
#           The concatenation of observation and noisy action.
#           Caller is responsible for concatenating before passing in.
#       t = [batch]
#           The current denoising timestep, one per sample.

        t_emb = self.time_emb(t)
#       Convert scalar timestep → 256-dim embedding.
#       Computed ONCE and reused by all 8 blocks.
#       Shape: [batch] → [batch, 256]

        x = self.input_proj(x)
#       Project input from 30 → 256 dimensions.
#       Shape: [batch, 30] → [batch, 256]

        for block in self.blocks:
            x = block(x, t_emb)
#       Run all 8 blocks in sequence. Each block:
#         - Takes the current x and the time embedding
#         - Returns an updated x (same shape)
#       x is progressively refined by each block.
#       t_emb is the same for all blocks — time doesn't change within one step.

        return self.output_proj(x)
#       Project from 256 → 10 dimensions.
#       Returns the predicted (cleaner) action.
#       Shape: [batch, 256] → [batch, 10]


# =============================================================================
# THE FULL FORWARD PASS — DATA FLOW
# =============================================================================
#
#   Input:  x=[batch,30], t=[batch]
#              │                │
#              │          TimeEmbedding
#              │                │ t_emb=[batch,256]
#         input_proj            │
#              │ [batch,256]    │
#              ▼                ▼
#         Block 1 ──── norm+linear+time_proj ────▶ x [batch,256]
#         Block 2 ──── norm+linear+time_proj ────▶ x [batch,256]
#         ...
#         Block 8 ──── norm+linear+time_proj ────▶ x [batch,256]
#              │
#         output_proj
#              │
#   Output: predicted_action [batch, 10]
#
# =============================================================================
# WHERE THIS FITS IN FlowRT — LATENCY BREAKDOWN
# =============================================================================
#
#   This forward pass runs ~20 times per robot decision (20 denoising steps).
#   At 100Hz control, you have 10ms total per decision → 0.5ms per step.
#
#   Current PyTorch: ~6ms per step (naive)     ← you measured this baseline
#   TensorRT FP16:   ~2ms per step
#   FlowRT target:   ~0.5ms per step
#
#   The 8 blocks are the hot path. Each block has 3 separate kernels today:
#     LayerNorm → write → read → Linear → write → read → time_proj → write
#   FlowRT fuses these into 1 kernel:
#     read x once → LayerNorm+Linear+time_proj → write once
#   That's the ln_linear_time.cu kernel (Milestone 1, Step 4).
# =============================================================================
