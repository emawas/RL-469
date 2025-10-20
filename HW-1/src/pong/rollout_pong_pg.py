import argparse
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
from src.pong.policy_cnn import PolicyCNN
from src.pong.preprocess import preprocess
import ale_py

def pick_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def rollout(env, policy, device, episodes=10, seed=0):
    """
    Run greedy rollouts with a trained Pong policy (no baseline).
    Returns an array of episode returns.
    """
    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        obs, _, _, _, _ = env.step(1)  # FIRE action to start Pong
        prev = None
        done = False
        total_r = 0

        while not done:
            frame = preprocess(obs)
            if prev is None:
                inp = frame
            else:
                inp = frame - prev
            prev = frame

            # Forward pass
            x = torch.as_tensor(inp, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            probs = policy(x)
            action = torch.argmax(probs, dim=-1).item()
            ale_action = 3 if action == 0 else 2  # match training mapping

            obs, r, terminated, truncated, _ = env.step(ale_action)
            done = terminated or truncated
            total_r += r

        returns.append(total_r)
        print(f"[ep {ep+1:4d}] return={total_r:6.1f}", flush=True)

    return np.array(returns, dtype=np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--results_dir", type=str, default="results/pong_pg_rollouts")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[eval] device: {device}")

    # Environment setup
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

    # Load trained policy
    policy = PolicyCNN(hidden=200).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt)
    policy.eval()
    print(f"[loaded] checkpoint: {args.checkpoint}")

    # Run rollouts
    returns = rollout(env, policy, device, episodes=args.episodes, seed=args.seed)

    # Save results
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "returns.npy", returns)
    print(f"[saved] returns -> {outdir/'returns.npy'}")

    # Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(returns, bins=20, color="seagreen", edgecolor="black", alpha=0.7)
    plt.title(f"Pong (No Baseline) — {args.episodes} Rollouts\nMean={returns.mean():.2f}, Std={returns.std():.2f}")
    plt.xlabel("Episode Return"); plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "returns_hist.png", dpi=150)
    plt.close()
    print(f"[plot saved] {outdir/'returns_hist.png'}")

    print(f"\nMean Return: {returns.mean():.3f}")
    print(f"Std Dev:     {returns.std():.3f}")

if __name__ == "__main__":
    main()