import argparse, os, csv
from pathlib import Path
import numpy as np
import torch, torch.optim as optim
import gymnasium as gym
from src.pong.preprocess import frame_input, preprocess
from src.pong.policy_cnn import PolicyCNN, select_action_sample
import ale_py



# ---------- Device ----------
def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        cap  = torch.cuda.get_device_capability(0)
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
    p.add_argument("--episodes", type=int, default=20000)      # Pong needs many episodes
    p.add_argument("--gamma", type=float, default=0.99)        # per prompt
    p.add_argument("--lr", type=float, default=1e-3)           # 0.01 is usually too high; note in report
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", type=str, default="results/pong")
    p.add_argument("--no_frame_diff", action="store_true", help="disable frame differencing")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env: Pong-v5. Use the ALE namespace to be explicit in Gymnasium.
    # Note: requires gymnasium[atari] and ale-py installed in your env.
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    
    obs, info = env.reset(seed=args.seed)
    # kick off a serve if needed
    obs, _, _, _, _ = env.step(1)   # 1 == FIRE (ALE)
    frame = preprocess(obs)
    # Policy + optimizer
    policy = PolicyCNN(hidden=args.hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Results dir
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
        # --- Logging setup ---
    rewards_csv = outdir / "rewards.csv"
    with open(rewards_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return", "loss"])

    with torch.no_grad():
        init_norm = sum(p.abs().sum().item() for p in policy.parameters())
    print(f"[init] |theta|_1={init_norm:.2f}")

    episode_returns = []
    policy.train()

    for ep in range(1, args.episodes + 1):
        # inside your episode loop
        logps = []
        rewards = []
        prev = None
        done = False
        
        obs, info = env.reset(seed=args.seed + ep)
        logps, rewards = [], []

        obs, _, _, _, _ = env.step(1)  # 1 == FIRE (ALE)
        while not done:
            frame = preprocess(obs)                    # (80,80) np float
            if prev is None:
                inp = frame
            else:
                inp = frame - prev                     # frame differencing
            prev = frame
        
            a, logp = select_action_sample(policy, inp, device=device)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
        
            logps.append(logp)                         # KEEP as tensor with grad
            rewards.append(float(r))
        # After the episode -----------------------------------------
        T = len(logps)
        if T == 0:
            print(f"[warn] ep {ep}: collected T=0 steps (skipping update)", flush=True)
            continue
        
        # If something weird made lengths mismatch, align them
        minT = min(len(logps), len(rewards))
        if minT != T or minT != len(rewards):
            print(f"[warn] ep {ep}: aligning lengths logps={len(logps)} rewards={len(rewards)} -> {minT}", flush=True)
        
        # after you collected `logps` (list of tensors) and `rewards` (list of floats)
        logps_t   = torch.stack(logps).to(device)                    # (T,)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)  # (T,)
        
        G  = compute_returns(rewards, gamma=args.gamma).to(device) # (T,)
        
        # variance reduction (only if T >= 2)
        if G.numel() >= 2:
            G = (G - G.mean()) / (G.std() + 1e-8)
        
        assert logps_t.requires_grad, "log-probs are detached"
        assert logps_t.shape == G.shape, f"{logps_t.shape} vs {G.shape}"
        
        loss = -(logps_t * G).mean()
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                                # <-- now has a grad_fn
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        print(f"[ep {ep:6d}] loss={loss.item():.6e}", flush=True)
        ep_ret = float(sum(rewards))
        episode_returns.append(ep_ret)

        with open(rewards_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_ret, float(loss.item())])

        if ep % 100 == 0 or ep == 1:
            ma100 = moving_average(episode_returns, k=100)
            ma_val = ma100[-1] if len(ma100) else np.nan
            with torch.no_grad():
                pnorm = sum(p.abs().sum().item() for p in policy.parameters())
            print(f"[ep {ep:6d}] ret={ep_ret:7.1f}  MA100={ma_val:7.2f}  loss={loss.item():.4f}  |theta|_1={pnorm:.1f}", flush=True)
        # --- Periodic checkpoint save ---
        if ep % 1000 == 0:
            ckpt_dir = outdir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(policy.state_dict(), ckpt_dir / f"policy_ep{ep}.pt")
            torch.save(policy.state_dict(), outdir / "policy_latest.pt")
            # Save and plot
            torch.save(policy.state_dict(), outdir / "policy.pt")

    try:
        import matplotlib.pyplot as plt
        xs = np.arange(1, len(episode_returns) + 1)
        ma100 = moving_average(episode_returns, k=100)
        plt.figure()
        plt.plot(xs, episode_returns, label="Episode Return", linewidth=1)
        if len(ma100) >= 1 and len(xs) >= 100:
            plt.plot(xs[99:], ma100, label="MA(100)", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Return")
        plt.title("ALE/Pong-v5 (γ=0.99) — REINFORCE (no baseline)")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "learning_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Plot failed:", e)

if __name__ == "__main__":
    main()