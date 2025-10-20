# src/pong/value_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueCNN(nn.Module):
    """Maps a preprocessed (1,80,80) Pong frame to a scalar V(s)."""
    def __init__(self, hidden: int = 200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 80, 80)
            feat  = self.conv(dummy)
            flat_dim = feat.view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        v = self.head(z)
        return v.squeeze(-1)  # (B,)