import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# pass path to rewards.csv as argument, e.g.:
# python plot_cartpole_returns.py results/cartpole_baseline_gpu_20251018_214610/rewards.csv
csv_path = Path(sys.argv[1])
df = pd.read_csv(csv_path)

episodes = df["episode"].to_numpy()
returns = df["return"].to_numpy()

def moving_average(x, k=100):
    x = np.asarray(x, float)
    if len(x) < k:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / float(k)

plt.figure(figsize=(7, 5))
plt.plot(episodes, returns, lw=0.8, label="Return (Baseline Agent)")
if len(returns) >= 100:
    plt.plot(episodes[99:], moving_average(returns, 100), lw=2, label="MA100")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("CartPole — REINFORCE with Baseline")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

out_png = csv_path.parent / "return_vs_episode_baseline.png"
plt.savefig(out_png, dpi=150)
plt.close()

print(f"✅ Saved plot to {out_png}")