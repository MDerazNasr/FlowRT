"""
export_unifolm_dit.py
---------------------
Export the unifolm-vla DiT action head to ONNX and measure baseline latency.

Mirrors export_diffusion_policy.py in structure.

This script demonstrates two things:
  1. The four export blockers in the original unifolm-vla codebase are resolved
     by SingleStepActionHead (see models/unifolm_vla/single_step.py).
  2. Even with a compiled ONNX model, the Python denoising loop still adds
     per-step overhead. This is what a C++ pipeline (RobotActionPipeline in
     OpenVINO GenAI) eliminates on Intel iGPU.
"""

import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from models.unifolm_vla import SingleStepActionHead

# --- Model config matching unifolm-vla DiT-L / Unitree G1 ---
STATE_DIM = 14              # Unitree G1 proprioception (14 DoF)
ACTION_DIM = 14             # 14-DoF joint actions
ACTION_HORIZON = 16         # Action chunk length
HIDDEN_SIZE = 1536          # DiT-L hidden dim
INPUT_EMBEDDING_DIM = 1536
NUM_LAYERS = 16             # DiT-L depth
NUM_ATTENTION_HEADS = 32
ATTENTION_HEAD_DIM = 48
CROSS_ATTENTION_DIM = 3584  # Qwen2.5-VL-7B hidden size
VL_SEQ_LEN = 512            # Typical VLM output sequence length
NUM_INFERENCE_STEPS = 10
NUM_TIMESTEP_BUCKETS = 1000


def build_model():
    return SingleStepActionHead(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        hidden_size=HIDDEN_SIZE,
        input_embedding_dim=INPUT_EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        attention_head_dim=ATTENTION_HEAD_DIM,
        cross_attention_dim=CROSS_ATTENTION_DIM,
    )


def export_to_onnx(model, path="unifolm_dit_single_step.onnx"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dummy inputs — torch.randn is OUTSIDE the model (Blocker 2 resolved)
    noisy_actions = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=device)
    state         = torch.randn(1, 1, STATE_DIM, device=device)
    vl_features   = torch.randn(1, VL_SEQ_LEN, CROSS_ATTENTION_DIM, device=device)
    timestep      = torch.zeros(1, dtype=torch.long, device=device)

    # Blocker 1: No loop inside model (loop is external)
    # Blocker 2: Noise tensor is an input argument, not generated inside model
    # Blocker 3: No torch.autocast("cuda") in forward path
    # Blocker 4: Plain tensor I/O, no BatchFeature

    torch.onnx.export(
        model,
        (noisy_actions, state, vl_features, timestep),
        path,
        input_names=["noisy_actions", "state", "vl_features", "timestep"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"},
            "state":         {0: "batch"},
            "vl_features":   {0: "batch"},
            "timestep":      {0: "batch"},
            "velocity":      {0: "batch"},
        },
        opset_version=17,
    )

    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export validated: {path}")
    print(f"Graph inputs:  {[n.name for n in onnx_model.graph.input]}")
    print(f"Graph outputs: {[n.name for n in onnx_model.graph.output]}")
    print(f"Graph nodes:   {len(onnx_model.graph.node)}")
    return path


def measure_baseline(model, n_steps=NUM_INFERENCE_STEPS, n_runs=20):
    """
    PyTorch baseline: N separate forward passes in a Python loop.
    This is the per-step dispatch overhead we target with the C++ pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    vl_features = torch.randn(1, VL_SEQ_LEN, CROSS_ATTENTION_DIM, device=device)
    state       = torch.randn(1, 1, STATE_DIM, device=device)

    # Warmup
    with torch.no_grad():
        actions = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=device)
        dt = 1.0 / n_steps
        for t in range(n_steps):
            t_disc = torch.tensor(
                [int(t / n_steps * NUM_TIMESTEP_BUCKETS)], dtype=torch.long, device=device
            )
            velocity = model(actions, state, vl_features, t_disc)
            actions = actions + dt * velocity

    latencies = []
    for _ in range(n_runs):
        actions = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            dt = 1.0 / n_steps
            for t in range(n_steps):
                t_disc = torch.tensor(
                    [int(t / n_steps * NUM_TIMESTEP_BUCKETS)], dtype=torch.long, device=device
                )
                velocity = model(actions, state, vl_features, t_disc)
                actions = actions + dt * velocity

        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"\nBaseline PyTorch — unifolm-vla DiT ({n_steps} steps, Euler, {device})")
    print(f"Mean: {latencies.mean():.2f} ms")
    print(f"Std:  {latencies.std():.2f} ms")
    print(f"Min:  {latencies.min():.2f} ms")
    print(f"Max:  {latencies.max():.2f} ms")
    print(f"Target for 25Hz control: <40ms per action chunk")


def measure_onnx(onnx_path, n_steps=NUM_INFERENCE_STEPS, n_runs=20):
    """
    ONNX Runtime: same Python loop, model loaded via ORT.

    Even with a compiled model, the Python loop adds per-step overhead
    (tensor allocation, session.run() call, numpy round-trip).
    This directly motivates the C++ denoising loop in RobotActionPipeline.
    """
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    vl_features = np.random.randn(1, VL_SEQ_LEN, CROSS_ATTENTION_DIM).astype(np.float32)
    state       = np.random.randn(1, 1, STATE_DIM).astype(np.float32)

    latencies = []
    for _ in range(n_runs):
        actions = np.random.randn(1, ACTION_HORIZON, ACTION_DIM).astype(np.float32)
        dt = 1.0 / n_steps

        t0 = time.perf_counter()
        for t in range(n_steps):
            t_disc = np.array(
                [int(t / n_steps * NUM_TIMESTEP_BUCKETS)], dtype=np.int64
            )
            velocity = session.run(
                ["velocity"],
                {
                    "noisy_actions": actions,
                    "state":         state,
                    "vl_features":   vl_features,
                    "timestep":      t_disc,
                },
            )[0]
            actions = actions + dt * velocity
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"\nONNX Runtime — unifolm-vla DiT ({n_steps} steps, Euler, Python loop)")
    print(f"Mean: {latencies.mean():.2f} ms")
    print(f"Std:  {latencies.std():.2f} ms")
    print(f"Min:  {latencies.min():.2f} ms")
    print(f"Note: Python loop overhead persists even with compiled model.")
    print(f"      A C++ denoising loop (RobotActionPipeline) eliminates this.")


if __name__ == "__main__":
    model = build_model()

    print("Exporting unifolm-vla DiT single step to ONNX...")
    onnx_path = export_to_onnx(model)

    print("\nMeasuring PyTorch baseline latency...")
    measure_baseline(model)

    print("\nMeasuring ONNX Runtime latency (Python loop)...")
    measure_onnx(onnx_path)
