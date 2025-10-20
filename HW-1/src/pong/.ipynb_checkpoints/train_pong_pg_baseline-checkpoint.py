import argparse, os, csv
from pathlib import Path
import numpy as np
import torch, torch.optim as optim
import gymnasium as gym
import ale_py

from src.pong.policy_cnn import PolicyCNN, select_action_sample
from src.pong.value_cnn import ValueCNN
from src.pong.preprocess import preprocess  # your 80x80 binary prepro

# ---------- Device ----------
def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr_pi", type=float, default=1e-3)
    ap.add_argument("--lr_v",  type=float, default=5e-3)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--results_dir", type=str, default="results/pong_pg_baseline")
    ap.add_argument("--no_frame_diff", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"[device] {device}")

    # Env
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)

    # Models & optimizers
    policy = PolicyCNN(hidden=args.hidden).to(device)
    valuef = ValueCNN(hidden=args.hidden).to(device)
    opt_pi = optim.Adam(policy.parameters(), lr=args.lr_pi)
    opt_v  = optim.Adam(valuef.parameters(), lr=args.lr_v)

    # Results
    outdir = Path(args.results_dir); outdir.mkdir(parents=True, exist_ok=True)
    rewards_csv = outdir / "rewards.csv"
    with open(rewards_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return", "loss_pi", "loss_v"])

    with torch.no_grad():
        pnorm = sum(p.abs().sum().item() for p in policy.parameters())
        vnorm = sum(p.abs().sum().item() for p in valuef.parameters())
    print(f"[init] |theta_pi|_1={pnorm:.1f} |theta_v|_1={vnorm:.1f}")

    ep_returns = []
    policy.train(); valuef.train()

    for ep in range(1, args.episodes+1):
        logps, rewards, frames = [], [], []
        done = False
        obs, info = env.reset(seed=args.seed + ep)
        # serve ball
        obs, _, _, _, _ = env.step(1)  # FIRE

        prev = None
        while not done:
            frame = preprocess(obs)           # (80,80) float in {0,1}
            inp   = frame if (prev is None or args.no_frame_diff) else (frame - prev)
            prev  = frame

            # sample action & store logp
            a, logp = select_action_sample(policy, inp, device=device)
            # env step
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            logps.append(logp)                 # (requires_grad)
            rewards.append(float(r))
            frames.append(inp)                 # keep inputs for critic

        # tensors on device
        T = len(rewards)
        if T == 0:
            print(f"[warn] ep {ep}: T=0, skip")
            continue

        G_full = compute_returns(rewards, gamma=args.gamma).to(device)   # (T,)
        logps_t = torch.stack(logps).to(device)                          # (T,)
        X = torch.as_tensor(np.stack(frames), dtype=torch.float32, device=device)  # (T,80,80)
        X = X.unsqueeze(1)  # (T,1,80,80)

        # Critic prediction on all steps
        V = valuef(X)                                                     # (T,)
        adv_full = G_full - V                                             # (T,)

        # Normalize advantages (variance reduction)
        if adv_full.numel() >= 2:
            adv_full = (adv_full - adv_full.mean()) / (adv_full.std() + 1e-8)

        # Actor update: OPTION A — only scoring timesteps (common for Pong)
        # mask = torch.tensor(rewards, device=device, dtype=torch.float32) != 0
        # if mask.sum() == 0:
        #     with open(rewards_csv, "a", newline="") as f:
        #         csv.writer(f).writerow([ep, float(sum(rewards)), 0.0, 0.0])
        #     continue
        # loss_pi = -(logps_t[mask] * adv_full.detach()[mask]).mean()

        # Actor update: OPTION B — use ALL timesteps (often still works)
        loss_pi = -(logps_t * adv_full.detach()).mean()

        # Critic loss on ALL steps
        loss_v  = 0.5 * (adv_full ** 2).mean()

        # Critic step
        opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(valuef.parameters(), 5.0)
        opt_v.step()

        # Actor step
        opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt_pi.step()

        ep_ret = float(sum(rewards))
        ep_returns.append(ep_ret)

        # Log row for live plotting
        with open(rewards_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_ret, float(loss_pi.item()), float(loss_v.item())])

        # Console log
        if ep % 100 == 0 or ep == 1:
            ma100 = moving_average(ep_returns, k=100)
            ma_val = ma100[-1] if len(ma100) else np.nan
            with torch.no_grad():
                pnorm = sum(p.abs().sum().item() for p in policy.parameters())
                vnorm = sum(p.abs().sum().item() for p in valuef.parameters())
            print(f"[ep {ep:6d}] ret={ep_ret:7.1f} MA100={ma_val:7.2f} "
                  f"loss_pi={loss_pi.item():.4f} loss_v={loss_v.item():.4f} "
                  f"|theta_pi|_1={pnorm:.1f} |theta_v|_1={vnorm:.1f}", flush=True)

        # periodic checkpoints (optional)
        if ep % 1000 == 0:
            ckpt_dir = outdir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({"policy": policy.state_dict(), "value": valuef.state_dict()},
                       ckpt_dir / f"ckpt_ep{ep}.pt")

    # final save
    torch.save({"policy": policy.state_dict(), "value": valuef.state_dict()},
               outdir / "checkpoint.pt")
    print(f"[done] saved to {outdir}")

if __name__ == "__main__":
    main()