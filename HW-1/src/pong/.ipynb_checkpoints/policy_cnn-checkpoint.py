# src/pong/policy_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# src/pong/policy_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyCNN(nn.Module):
    def __init__(self, hidden: int = 200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
        )

        # Infer flatten dim from the actual input size (80x80 after your preprocess)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 80, 80)
            feat = self.conv(dummy)
            flat_dim = feat.view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # actions: LEFT / RIGHT
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        logits = self.head(z)
        return F.softmax(logits, dim=-1)

def select_action_sample(policy, frame, device="cpu"):
    """
    Given a preprocessed Pong frame (80x80), select an action according to the current policy.
    Returns:
        ale_action (int): 2 for RIGHT, 3 for LEFT (the only two we train on)
        logp (Tensor): log probability of the chosen action (requires grad)
    """
    # Convert frame to tensor (expect 80x80 np array)
    x = torch.as_tensor(frame, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    # shape = (1, 1, 80, 80)

    # Forward pass through CNN — NO torch.no_grad(), we need gradients
    probs = policy(x)                            # (1, 2)
    dist  = torch.distributions.Categorical(probs=probs)

    # Sample action in {0,1}
    a01   = dist.sample()

    # Log-probability of that action (keep gradient)
    logp  = dist.log_prob(a01).squeeze(0)        # scalar tensor, requires_grad=True

    # Map 0→LEFT(3), 1→RIGHT(2) — standard Pong mapping
    ale_action = 3 if int(a01.item()) == 0 else 2

    return ale_action, logp

def select_action_greedy(policy: "PolicyCNN", obs_img, device="cpu"):
    x = torch.as_tensor(obs_img, dtype=torch.float32, device=device)
    # Normalize shape to (N, C, H, W) = (1, 1, 80, 80)
    if x.ndim == 2:                 # (80,80)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:               # (C,H,W) or (H,W,C)
        # assume (1,80,80); if (80,80,1) permute to (1,80,80)
        if x.shape[-1] in (1, 2, 3) and x.shape[0] == 80:
            x = x.permute(2,0,1)    # (C,H,W)
        x = x.unsqueeze(0)          # (1,C,H,W)
    elif x.ndim == 4:               # already batched
        pass
    elif x.ndim == 5:               # e.g. (1,1,1,80,80) -> squeeze the extra singleton
        x = x.squeeze(1)
    else:
        raise ValueError(f"Unexpected obs shape {tuple(x.shape)}")

    probs = policy(x)                     # (N,2)
    action = torch.argmax(probs, dim=-1)  # greedy for eval
    # map 0->2 (RIGHT), 1->3 (LEFT)
    return 2 + int(action.item())

