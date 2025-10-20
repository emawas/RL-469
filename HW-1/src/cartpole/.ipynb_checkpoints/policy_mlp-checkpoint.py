# src/cartpole/policy_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)  # outputs logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)  # normalize over actions
        return probs

def select_action_sample(policy: "Policy", obs, device="cpu"):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    probs = policy(obs_t)  # softmax output
    dist  = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    logp = dist.log_prob(action).squeeze(0)   # <-- ensure scalar ( )
    return int(action.item()), logp

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)
def select_action_sample(policy: "Policy", obs, device="cpu"):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    probs = policy(obs_t)  # softmax output
    dist  = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    logp = dist.log_prob(action).squeeze(0)   # <-- ensure scalar ( )
    return int(action.item()), logp