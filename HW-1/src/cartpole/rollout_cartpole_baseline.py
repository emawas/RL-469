import argparse, csv
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from src.cartpole.policy_mlp import Policy  # same Policy you trained

def pick_device(dev_str):
    if dev_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def run_episode(env, policy, device, seed=None, greedy=True):
    obs, info = env.reset(seed=seed)
    done = False
    ep_ret = 0.0
    with torch.no_grad():
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            probs = policy(obs_t)
            if greedy:
                a = int(torch.argmax(probs, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs=probs)
                a = int(dist.sample().item())
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)
            done = terminated or truncated
    return ep_ret

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint.pt from baseline trainer")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--results_dir", type=str, default="results/cartpole_baseline_eval")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of greedy")
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[eval] device: {device}")

    # Load env
    env = gym.make("CartPole-v1")

    # Build policy and load weights from baseline checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # We don’t need the value net for rollout; policy only.
    # Infer obs/act dims from env
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    from src.cartpole.policy_mlp import Policy
    # NOTE: this Policy expects logits->softmax in forward (as in your code)
    policy = Policy(obs_dim=obs_dim, act_dim=act_dim, hidden=args.hidden).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # Rollouts
    returns = []
    for i in range(args.episodes):
        ret = run_episode(env, policy, device, seed=args.seed + i, greedy=not args.stochastic)
        returns.append(ret)
        if (i+1) % 50 == 0:
            print(f"[eval] {i+1}/{args.episodes} episodes done (last ret={ret:.1f})", flush=True)

    env.close()

    # Stats
    rets = np.array(returns, dtype=float)
    mean = float(rets.mean())
    std  = float(rets.std(ddof=0))
    print(f"[eval] mean={mean:.3f}  std={std:.3f}")

    # Save outputs
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = outdir / "rollout_returns.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode","return"])
        for i, r in enumerate(returns, 1):
            w.writerow([i, r])

    # Histogram
    plt.figure(figsize=(7,5))
    plt.hist(rets, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Episode Return")
    plt.ylabel("Count")
    plt.title(f"CartPole-v1 (baseline) — {args.episodes} rollouts\nmean={mean:.2f}, std={std:.2f}")
    plt.tight_layout()
    hist_path = outdir / "rollout_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print(f"[eval] wrote: {csv_path}")
    print(f"[eval] wrote: {hist_path}")

if __name__ == "__main__":
    main()