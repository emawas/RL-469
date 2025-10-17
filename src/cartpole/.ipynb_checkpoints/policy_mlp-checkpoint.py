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
    # Convert observation to a tensor and send to the correct device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Forward pass → now returns probabilities (because of softmax)
    probs = policy(obs_t)
    
    # Define a categorical distribution over actions using probabilities
    dist = torch.distributions.Categorical(probs=probs)
    
    # Sample an action from the policy distribution
    action = dist.sample()
    
    # Log probability of the chosen action (used in REINFORCE update)
    logp = dist.log_prob(action)
    
    # Return both: 
    # - the sampled action (as an integer for env)
    # - the log probability (as a tensor for gradient computation)
    return int(action.item()), logp