# -*- coding: utf-8 -*-
import os, re, glob, argparse, time, random, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

# -------------------------
# Args
# -------------------------
p = argparse.ArgumentParser()
p.add_argument("--run_dir", type=str, default="results/cartpole_dqn")
p.add_argument("--ckpt", type=str, default=None, help="Optional explicit checkpoint path")
p.add_argument("--episodes", type=int, default=500)
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--seed", type=int, default=-1, help="-1 => randomize base seed")
p.add_argument("--deterministic", action="store_true",
               help="Force deterministic eval (sets seed=123)")
args = p.parse_args()

device = torch.device(args.device)

# -------------------------
# Env
# -------------------------
if args.deterministic:
    base_seed = 123
else:
    base_seed = args.seed if args.seed >= 0 else (int(time.time_ns() ^ os.getpid() ^ random.getrandbits(32)))

env = gym.make("CartPole-v1")
print(f"[eval] device={device}  base_seed={base_seed}  deterministic={args.deterministic}")

# -------------------------
# Net (match training)
# -------------------------
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# -------------------------
# Load latest (or provided) checkpoint
# -------------------------
ckpt_path = args.ckpt
if ckpt_path is None:
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    def _ep_num(path):
        m = re.search(r"ckpt_ep(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_ep*.pt")), key=_ep_num)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    ckpt_path = ckpts[-1]

print(f"[eval] Loading checkpoint: {ckpt_path}")

# Build net with env dims
obs0, _ = env.reset(seed=base_seed)
n_obs = len(obs0)
n_actions = env.action_space.n

policy = DQN(n_obs, n_actions).to(device)
state_dict = torch.load(ckpt_path, map_location=device)
policy.load_state_dict(state_dict)   # your training saved raw state_dict
policy.eval()

@torch.no_grad()
def greedy_action(state_tensor):
    q = policy(state_tensor)                 # [1, n_actions]
    return int(q.argmax(1).item())

def run_episode(ep_index: int) -> float:
    obs, _ = env.reset(seed=base_seed + ep_index)
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    total_r = 0.0
    done = False
    while not done:
        a = greedy_action(state)
        obs_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total_r += float(r)
        if not done:
            state = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
    return total_r

# Seed global RNGs only if deterministic or explicit seed provided
if args.deterministic or args.seed >= 0:
    random.seed(base_seed); np.random.seed(base_seed); torch.manual_seed(base_seed)

# -------------------------
# Evaluate
# -------------------------
returns = []
for i in range(args.episodes):
    ret = run_episode(i)
    returns.append(ret)
    if (i + 1) % 10 == 0:
        print(f"[eval] ep {i+1}/{args.episodes}  return={ret:.1f}", flush=True)

returns = np.asarray(returns, dtype=np.float32)
print("\n=== Evaluation summary (CartPole-v1) ===")
print(f"episodes: {len(returns)}")
print(f"mean:   {returns.mean():.2f}")
print(f"median: {np.median(returns):.2f}")
print(f"std:    {returns.std():.2f}")
print(f"min:    {returns.min():.2f}")
print(f"max:    {returns.max():.2f}")

# -------------------------
# Plot & save histogram
# -------------------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(7,4))
plt.hist(returns, bins=30, edgecolor="black", alpha=0.8)
plt.xlabel("Episode return")
plt.ylabel("Count")
title_stem = os.path.basename(ckpt_path)
plt.title(f"CartPole — {title_stem} — {len(returns)} eps\nbase_seed={base_seed}")
plt.tight_layout()
out_png = os.path.join(args.run_dir, f"eval_hist_{len(returns)}eps_{ts}.png")
os.makedirs(args.run_dir, exist_ok=True)
plt.savefig(out_png, dpi=150)
print(f"[eval] Saved histogram: {out_png}")