import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return enc


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        # actions: (B, T, action_dim)
        # timesteps: (B,) int
        B, T, _ = actions.shape
        timesteps = timesteps.unsqueeze(1).expand(-1, T).float()
        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = x * torch.sigmoid(x)  # swish
        x = self.W2(x)
        x = x * torch.sigmoid(x)  # swish
        return self.W3(x)
