import torch
import torch.nn as nn
import onnx  # lib that lets us inspect/validate ONNX graph files after export
# the runtime that loads an ONNX file and runs inference on it. This is the engine we wrap with custom ops
import onnxruntime as ort
import numpy as np
import time
from torch.profiler import profile, record_function, ProfilerActivity


'''
building model to mimic diffusion policy weights:
- layer types
  - input/output shapes
- time conditioning structure
- The profiling results will be identical because the bottlenecks come from the architecture
Real weights are large and require huggingface login

'''

# every pytorch model inherits from nn.Modulke
# need to call super in every constructor, it sets up internal Pytorch bookkeeping otherwise things would break


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #   A fully connected layer. Stores a weight matrix W
        # and bias b. When called: output = input @ W.T + b.
        # That is a GEMM. Every nn.Linear in this file is the
        # same operation you just wrote in CUDA
        self.linear1 = nn.Linear(1, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        t = torch.relu(self.linear1(t))
        return self.linear2(t)


class DiffusionPolicyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x, t_emb):
        return self.linear(self.norm(x)) + self.time_proj(t_emb)


class DiffusionPolicyNet(nn.Module):
    def __init__(self, obs_dim=20, action_dim=10, hidden_dim=256, n_layers=8):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.time_emb = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [DiffusionPolicyBlock(hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, t_emb)
        return self.output_proj(x)


'''
  Three things to understand before moving on.

  TimeEmbedding takes the scalar timestep t and
  projects it into a vector. This vector gets added
  into every block so the model always knows where it
  is in the denoising trajectory. This is the time
  conditioning that makes the activations look
  different at t=0 vs t=1 — the core motivation for
  time-conditioned quantization in Milestone 2.

  DiffusionPolicyBlock is one layer. Notice it does
  LayerNorm, then Linear, then adds the time
  embedding. Those three operations — norm, linear,
  time injection — are exactly what our fused kernel
  on Day 4 will collapse into a single GPU operation.

  n_layers=8 means 8 blocks. Each forward pass runs
  all 8. The full inference loop runs this 50 times.
  That is 400 block executions per sample.
'''

# Export to onnx and validate


def export_to_onnx(model, obs_dim=20, action_dim=10, path="diffusion_policy.onnx"):
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)

    # represenattive inputs: batch=1, one observation, one noisy action, one timestep
    x_dummy = torch.randn(1, obs_dim + action_dim, device=device)
    t_dummy = torch.zeros(1, device=device)

    torch.onnx.export(
        model,
        (x_dummy, t_dummy),
        path,
        input_names=["x", "t"],
        output_names=["velocity"],
        dynamic_axes={"x": {0: "batch"}, "t": {
            0: "batch"}, "velocity": {0: "batch"}},
        opset_version=17,
    )

    # validate the graph is well-formed
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export validated: {path}")
    print(f"Graph inputs:{[n.name for n in onnx_model.graph.input]}")
    print(f"Graph outputs:{[n.name for n in onnx_model.graph.output]}")
    print(f"Graph nodes:{len(onnx_model.graph.node)}")

    '''
 dynamic_axes tells the exporter that the batch
  dimension can vary at runtime. Without this, the
  ONNX graph is frozen to batch=1 and will reject any
  other size. We always set this even when we only
  ever use batch=1 — it is good practice.

  onnx.checker.check_model validates that the graph is
   well-formed — all shapes are consistent, all ops
  are valid. This catches export errors early before
  you waste time trying to run a broken graph. The
  number of graph nodes it prints will be larger than
  8 — ONNX breaks each PyTorch operation into
  primitive ops.
    '''

# baseline inference loop and latency measurement


def measure_baseline(model, obs_dim=20, action_dim=10, n_steps=50, n_runs=20):
    device = torch.device("cuda")
    model = model.to(device).eval()

    x = torch.randn(1, obs_dim + action_dim, device=device)
    timesteps = torch.linspace(0, 1, n_steps, device=device)

    # warm up
    with torch.no_grad():
        for t in timesteps:
            v = model(x, t.unsqueeze(0))
            x[:, obs_dim:] = x[:, obs_dim:] + v * (1.0 / n_steps)

    # timed runs
    latencies = []
    for _ in range(n_runs):
        x = torch.randn(1, obs_dim + action_dim, device=device)
        # same reason we use cudaDeviceSynchronize()
        # GPU launches are asynchronous
        # if you dont synchronize before and after
        # you would be measuring CPU scheduling time not GPU execution time
        #
        #
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for t in timesteps:
                v = model(x, t.unsqueeze(0))
                x[:, obs_dim:] = x[:, obs_dim:] + v * (1.0 / n_steps)

        torch.cuda.synchronize()
        # per_counter gives nanosecond resolution on linux
        # use over time.time() for precision
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)

    print(f"\nBaseline PyTorch inference (50 steps,Euler)")
    print(f"Mean:{latencies.mean():.2f} ms")
    print(f"Std:{latencies.std():.2f} ms")
    print(f"Min:{latencies.min():.2f} ms")
    print(f"Max:{latencies.max():.2f} ms")
    print(f"Target: <10ms for 100Hz robot control")
    # we run 20 repetetions
    # we report mean, std, min, max
    # variance tells you whether GPU is thermal throttling or being interupted by other processes
    # high std relatibe to mean is warning that measurements are noisy
    #

# Fuction to profile missing dependencies on gpu


def profile_inference(model, obs_dim=20, action_dim=10, n_steps=50):
    device = torch.device("cuda")
    model = model.to(device).eval()

    x = torch.randn(1, obs_dim + action_dim, device=device)
    timesteps = torch.linspace(0, 1, n_steps, device=device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False
    ) as prof:
        with torch.no_grad():
            for t in timesteps:
                with record_function(f"step"):
                    v = model(x, t.unsqueeze(0))
                    x[:, obs_dim:] = x[:, obs_dim:] + v * (1.0 / n_steps)

    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))


if __name__ == "__main__":
    model = DiffusionPolicyNet(
        obs_dim=20,
        action_dim=10,
        hidden_dim=256,
        n_layers=8
    ).cuda()

    print("Exporting to ONNX...")
    export_to_onnx(model)

    print("\nMeasuring baseline latency...")
    measure_baseline(model)

    print("\nProfiling...")
    profile_inference(model)
