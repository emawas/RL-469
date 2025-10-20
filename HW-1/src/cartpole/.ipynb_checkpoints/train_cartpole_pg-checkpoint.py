import argparse, os, csv
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from pathlib import Path
from src.cartpole.policy_mlp import Policy, select_action_sample

# ---------- Device Picker ----------
def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        build = torch.version.cuda or "?"
        print(f"[device] CUDA build {build} | {name} | capability {cap}")
        return dev
    print("[device] CUDA not available -> CPU")
    return torch.device("cpu")

device = pick_device()
print(f"Using device: {device}")

# ---------- Helpers ----------
def moving_average(x, k=100):
    if len(x) < 1: return np.array([])
    x = np.asarray(x, dtype=float)
    if len(x) < k: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / float(k)

def compute_returns(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", type=str, default="results/cartpole")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env setup
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=args.seed)
    obs_dim = len(obs)
    act_dim = env.action_space.n  # 2 actions

    # Policy and optimizer
    policy = Policy(obs_dim=obs_dim, act_dim=act_dim, hidden=args.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Output dir
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rewards_csv = outdir / "rewards.csv"

    with open(rewards_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return"])

    with torch.no_grad():
        init_norm = sum(p.abs().sum().item() for p in policy.parameters())
    print(f"[init] |theta|_1={init_norm:.4f}")

    episode_returns = []

    for ep in range(1, args.episodes + 1):
        logps, rewards = [], []
        obs, info = env.reset()
        done = False

        while not done:
            # Sample action from softmax policy
            a, logp = select_action_sample(policy, obs, device=device)
            next_obs, r, terminated, truncated, info = env.step(a)

            logps.append(logp.to(device))
            rewards.append(float(r))
            obs = next_obs
            done = terminated or truncated

        # Compute discounted returns
        G = compute_returns(rewards, gamma=args.gamma).to(device)
        if len(G) >= 2:
            G = (G - G.mean()) / (G.std() + 1e-8)  # baseline-like normalization

        
        logps_t = torch.stack(logps)      # shape (T,)
        assert logps_t.ndim == 1 and G.ndim == 1 and logps_t.shape[0] == G.shape[0], \
               f"Bad shapes: logps_t {logps_t.shape}, G {G.shape}"
        
        loss = -(logps_t * G).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        ep_ret = sum(rewards)
        episode_returns.append(ep_ret)

        with open(rewards_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_ret])

        if ep % 100 == 0 or ep == 1:
            ma100 = moving_average(episode_returns, k=100)
            ma_val = ma100[-1] if len(ma100) >= 1 else np.nan
            with torch.no_grad():
                pnorm = sum(p.abs().sum().item() for p in policy.parameters())
            print(f"[ep {ep:5d}] ret={ep_ret:6.1f}  MA100={ma_val:6.1f}  loss={loss.item():.4f}  |theta|_1={pnorm:.1f}", flush=True)

    # Save policy
    torch.save(policy.state_dict(), outdir / "policy.pt")

    # Plot learning curve
    try:
        import matplotlib.pyplot as plt
        xs = np.arange(1, len(episode_returns) + 1)
        ma100 = moving_average(episode_returns, k=100)
        plt.figure()
        plt.plot(xs, episode_returns, label="Episode Return", linewidth=1)
        if len(ma100) >= 1:
            plt.plot(xs[99:], ma100, label="MA(100)", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Return")
        plt.title("CartPole-v1 (γ=0.95) — Policy Gradient w/ Softmax Policy")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "learning_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Plot failed:", e)

if __name__ == "__main__":
    main()