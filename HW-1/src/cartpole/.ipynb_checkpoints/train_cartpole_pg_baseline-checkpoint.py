# src/cartpole/train_cartpole_pg_baseline.py
import argparse, os, csv
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from pathlib import Path

from src.cartpole.policy_mlp import Policy, select_action_sample
from src.cartpole.value_mlp import ValueNet

def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr_pi", type=float, default=1e-3)
    p.add_argument("--lr_v", type=float, default=5e-3)   # critic can learn faster
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", type=str, default="results/cartpole_baseline")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Using device: {device}")

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=args.seed)
    obs_dim = len(obs)
    act_dim = env.action_space.n

    policy = Policy(obs_dim=obs_dim, act_dim=act_dim, hidden=args.hidden).to(device)
    valuef = ValueNet(obs_dim=obs_dim, hidden=args.hidden).to(device)

    opt_pi = optim.Adam(policy.parameters(), lr=args.lr_pi)
    opt_v  = optim.Adam(valuef.parameters(), lr=args.lr_v)

    outdir = Path(args.results_dir); outdir.mkdir(parents=True, exist_ok=True)
    rewards_csv = outdir / "rewards.csv"
    with open(rewards_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return"])

    with torch.no_grad():
        init_norm = sum(p.abs().sum().item() for p in policy.parameters())
        init_vnorm = sum(p.abs().sum().item() for p in valuef.parameters())
    print(f"[init] |theta_pi|_1={init_norm:.2f}  |theta_v|_1={init_vnorm:.2f}")

    episode_returns = []

    for ep in range(1, args.episodes + 1):
        logps, rewards, states = [], [], []

        obs, info = env.reset(seed=args.seed + ep)
        done = False

        while not done:
            states.append(np.asarray(obs, dtype=np.float32))
            a, logp = select_action_sample(policy, obs, device=device)

            next_obs, r, terminated, truncated, info = env.step(a)
            logps.append(logp)                  # (requires_grad=True)
            rewards.append(float(r))
            obs = next_obs
            done = terminated or truncated

        # Tensors
        G = compute_returns(rewards, gamma=args.gamma).to(device)   # (T,)
        S = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)  # (T, obs_dim)
        logps_t = torch.stack(logps).to(device)                     # (T,)

        # Critic: predict V(s)
        V = valuef(S)                                               # (T,)
        advantages = G - V                                          # (T,)

        # Optional: normalize advantages (helps variance; baseline still learned)
        if advantages.numel() >= 2:
            adv_mean, adv_std = advantages.mean(), advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Losses
        loss_pi = -(logps_t * advantages.detach()).mean()           # actor (REINFORCE with baseline)
        loss_v  = 0.5 * (advantages ** 2).mean()                     # critic MSE to returns

        # Update critic
        opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(valuef.parameters(), 5.0)
        opt_v.step()

        # Update actor
        opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt_pi.step()

        ep_ret = float(sum(rewards))
        episode_returns.append(ep_ret)

        with open(rewards_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_ret])

        if ep % 50 == 0 or ep == 1:
            ma100 = moving_average(episode_returns, k=100)
            ma_val = ma100[-1] if len(ma100) else np.nan
            with torch.no_grad():
                pnorm = sum(p.abs().sum().item() for p in policy.parameters())
                vnorm = sum(p.abs().sum().item() for p in valuef.parameters())
            print(f"[ep {ep:5d}] ret={ep_ret:6.1f} MA100={ma_val:6.1f} "
                  f"loss_pi={loss_pi.item():.4f} loss_v={loss_v.item():.4f} "
                  f"|theta_pi|_1={pnorm:.1f} |theta_v|_1={vnorm:.1f}",
                  flush=True)

    torch.save({"policy": policy.state_dict(),
                "value": valuef.state_dict()},
               outdir / "checkpoint.pt")
    print(f"Saved to {outdir}")

if __name__ == "__main__":
    main()