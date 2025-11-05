# -*- coding: utf-8 -*-
import os, re, glob, argparse, random, time, math, datetime
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

# -------------------------
# Args
# -------------------------
p = argparse.ArgumentParser()
p.add_argument("--run_dir", type=str, default="results/mspacman_dqn1")
p.add_argument("--ckpt", type=str, default=None, help="Optional explicit checkpoint path")
p.add_argument("--episodes", type=int, default=500)
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--frameskip", type=int, default=4)
p.add_argument("--repeat_action_probability", type=float, default=0.25,
               help="Sticky actions; 0.0 = deterministic (training used 0.0)")
p.add_argument("--seed", type=int, default=-1,
               help="Base seed. -1 => randomize (different every run).")
p.add_argument("--noop_max", type=int, default=30, help="Random NOOPs at start (0 disables)")
p.add_argument("--deterministic", action="store_true",
               help="Force deterministic eval (overrides sticky/noop, fixed seed=123)")
args = p.parse_args()

device = torch.device(args.device)
gym.register_envs(ale_py)  # ensure ALE namespace is registered

# -------------------------
# Env (defaults aim for stochastic rollouts unless --deterministic)
# -------------------------
if args.deterministic:
    rep_prob = 0.0
    base_seed = 123
    noop_max = 0
else:
    rep_prob = args.repeat_action_probability
    # randomize seeds by default so runs differ every time
    base_seed = (args.seed if args.seed >= 0
                 else int(time.time_ns() ^ os.getpid() ^ random.getrandbits(32)))
    noop_max = args.noop_max

env = gym.make(
    "ALE/MsPacman-v5",
    obs_type="rgb",
    frameskip=args.frameskip,
    repeat_action_probability=rep_prob,
    render_mode=None,
)

print(f"[eval] device={device}  frameskip={args.frameskip}  sticky={rep_prob}  "
      f"noop_max={noop_max}  base_seed={base_seed}")
try:
    print("[eval] action meanings:", env.unwrapped.get_action_meanings())
except Exception:
    pass

# -------------------------
# Preprocess (match training)
# -------------------------
MSPACMAN_COLOR_SUM = 210 + 164 + 74

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    # crop/downsample to (88,80), grayscale int8 in [-128,127], shape (88,80,1)
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2)
    img[img == MSPACMAN_COLOR_SUM] = 0
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)

# -------------------------
# Network (match training)
# -------------------------
class DQN(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 88, 80)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            flat_dim = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_dim, 512)
        self.out = nn.Linear(512, outputs)

    def forward(self, x):
        x = x.to(device).float() / 128.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.out(x)

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

n_actions = env.action_space.n
policy = DQN(n_actions).to(device)
state = torch.load(ckpt_path, map_location=device, weights_only=False)
# accept either raw state_dict or {'state_dict': ...}
if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
    state = state["state_dict"]
policy.load_state_dict(state)
policy.eval()

@torch.no_grad()
def select_action(state_4chw: torch.Tensor) -> int:
    q = policy(state_4chw)  # [1, n_actions]
    return int(q.argmax(1).item())

def do_random_noops(max_noops: int):
    """Execute a random number of NOOPs (0..max_noops) to diversify starts."""
    if max_noops <= 0:
        return
    noop_action = 0  # MsPacman NOOP is 0 in ALE
    k = np.random.randint(0, max_noops + 1)
    for _ in range(k):
        obs, _, term, trunc, _ = env.step(noop_action)
        if term or trunc:
            break

# -------------------------
# Single episode rollout (raw return)
# -------------------------
def run_episode(ep_index: int) -> float:
    obs, _info = env.reset(seed=base_seed + ep_index)

    if not args.deterministic and noop_max > 0:
        do_random_noops(noop_max)

    proc = preprocess_observation(obs)
    frames = [proc, proc, proc, proc]                     # 4-frame stack
    stacked = np.concatenate(frames, axis=2)              # (88,80,4)
    state = torch.from_numpy(np.transpose(stacked, (2, 0, 1))).unsqueeze(0).to(device).float()

    total_raw = 0.0
    done = False
    while not done:
        action = select_action(state)
        next_obs, raw_r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_raw += float(raw_r)

        next_proc = preprocess_observation(next_obs)
        frames.pop(0); frames.append(next_proc)
        stacked = np.concatenate(frames, axis=2)
        state = torch.from_numpy(np.transpose(stacked, (2, 0, 1))).unsqueeze(0).to(device).float()
    return total_raw

# -------------------------
# Evaluate
# -------------------------
# Seed numpy/python/torch for reproducibility when user provided a fixed seed
if args.deterministic or args.seed >= 0:
    random.seed(base_seed); np.random.seed(base_seed); torch.manual_seed(base_seed)

returns = []
for i in range(args.episodes):
    ret = run_episode(i)
    returns.append(ret)
    if (i + 1) % 10 == 0:
        print(f"[eval] ep {i+1}/{args.episodes}  return={ret:.1f}", flush=True)

returns = np.asarray(returns, dtype=np.float32)
print("\n=== Evaluation summary (raw MsPacman score) ===")
print(f"episodes: {len(returns)}")
print(f"mean:   {returns.mean():.2f}")
print(f"median: {np.median(returns):.2f}")
print(f"std:    {returns.std():.2f}")
print(f"min:    {returns.min():.2f}")
print(f"max:    {returns.max():.2f}")

# -------------------------
# Plot & save histogram (timestamped)
# -------------------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(7,4))
plt.hist(returns, bins=30, edgecolor="black", alpha=0.8)
plt.xlabel("Episode return (raw Atari score)")
plt.ylabel("Count")
title_stem = os.path.basename(ckpt_path)
det_tag = ("deterministic"
           if args.deterministic
           else f"sticky={rep_prob}, noop_max={noop_max}, base_seed={base_seed}")
plt.title(f"MsPacman — {title_stem} — {len(returns)} eps\n{det_tag}")
plt.tight_layout()
out_png = os.path.join(args.run_dir, f"eval_hist_{len(returns)}eps_{ts}.png")
plt.savefig(out_png, dpi=150)
print(f"[eval] Saved histogram: {out_png}")