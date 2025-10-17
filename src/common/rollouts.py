import argparse
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

# --- import your softmax policy + greedy ---
from src.cartpole.policy_mlp import Policy

@torch.no_grad()
def select_action_greedy(policy: "Policy", obs, device="cpu"):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    probs = policy(obs_t)                       # policy returns probs
    a = torch.argmax(probs, dim=-1).item()      # greedy over probs
    return int(a)

def eval_cartpole(checkpoint: str, episodes: int, seed: int, results_dir: str, device="cpu"):
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=seed)
    obs_dim = len(obs)
    act_dim = env.action_space.n

    policy = Policy(obs_dim, act_dim, hidden=128).to(device)  # hidden doesn’t matter if weights loaded
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()

    rets = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        done, ep_ret = False, 0.0
        while not done:
            a = select_action_greedy(policy, obs, device=device)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)
            done = terminated or truncated
        rets.append(ep_ret)

    rets = np.asarray(rets, dtype=float)
    mean, std = rets.mean(), rets.std()

    outdir = Path(results_dir); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "eval_stats.txt", "w") as f:
        f.write(f"episodes: {episodes}\nmean: {mean:.4f}\nstd: {std:.4f}\n")

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(rets, bins=30)
        plt.title(f"CartPole-v1 — {episodes} episode returns\nmean={mean:.2f}, std={std:.2f}")
        plt.xlabel("Episode Return"); plt.ylabel("Count"); plt.tight_layout()
        plt.savefig(outdir / "histogram_episodes.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Plot failed:", e)

    print(f"[eval] episodes={episodes} mean={mean:.2f} std={std:.2f}")
    return mean, std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True, choices=["CartPole-v1"])
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--results_dir", type=str, default="results/cartpole_eval")
    ap.add_argument("--device", type=str, default="cpu")  # "cpu" or "cuda"
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.env == "CartPole-v1":
        eval_cartpole(args.checkpoint, args.episodes, args.seed, args.results_dir, device=device)

if __name__ == "__main__":
    main()