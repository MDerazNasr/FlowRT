import torch
import torch.nn as nn

from .dit import DiT
from .action_encoder import ActionEncoder


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


class SingleStepActionHead(nn.Module):
    """
    Single denoising step for the unifolm-vla flow matching action head.

    This is the ONNX-exportable unit. The Euler denoising loop lives outside
    this class — in a Python inference runner or a C++ pipeline (the OpenVINO
    RobotActionPipeline equivalent for Intel iGPU).

    Export blockers resolved vs the original unifolm-vla codebase:
      1. No Python for-loop (loop is external — no graph unrolling)
      2. No torch.randn inside the model (noise is an input argument)
      3. No torch.autocast("cuda") in forward path
      4. No BatchFeature API boundary (plain tensor I/O throughout)

    Inputs
    ------
    noisy_actions : (B, action_horizon, action_dim)   float
    state         : (B, 1, state_dim)                 float
    vl_features   : (B, S, cross_attention_dim)       float  -- VLM backbone output
    timestep      : (B,)                              long   -- discrete step index

    Output
    ------
    velocity : (B, action_horizon, action_dim)        float
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_size: int,
        input_embedding_dim: int,
        num_layers: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.state_encoder = MLP(state_dim, hidden_size, input_embedding_dim)
        self.action_encoder = ActionEncoder(action_dim, input_embedding_dim)
        self.dit = DiT(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            output_dim=hidden_size,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
        )
        self.action_decoder = MLP(hidden_size, hidden_size, action_dim)
        self.position_embedding = nn.Embedding(max_seq_len, input_embedding_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        state: torch.Tensor,
        vl_features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Encode state token
        state_features = self.state_encoder(state)              # (B, 1, D)

        # Encode noisy actions conditioned on timestep
        action_features = self.action_encoder(noisy_actions, timestep)  # (B, T, D)

        # Positional embedding over action tokens
        pos_ids = torch.arange(
            action_features.shape[1], dtype=torch.long, device=action_features.device
        )
        action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        # Concatenate: state token + action tokens
        sa_embs = torch.cat([state_features, action_features], dim=1)  # (B, 1+T, D)

        # DiT: cross-attention to VLM features, AdaLayerNorm per block
        model_output = self.dit(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_features,
            timestep=timestep,
        )

        # Decode and slice action tokens
        velocity = self.action_decoder(model_output)            # (B, 1+T, action_dim)
        return velocity[:, -self.action_horizon:]               # (B, T, action_dim)
