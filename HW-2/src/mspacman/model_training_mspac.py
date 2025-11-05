# -*- coding: utf-8 -*-
import gymnasium as gym
import math, random, matplotlib, csv, shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

from ale_py import ALEInterface

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================
# 1. Environment setup
# ============================================================

ale = ALEInterface()

env = gym.make(
    "ALE/MsPacman-v5",
    obs_type="rgb",
    frameskip=4,                   # like classic Gym
    repeat_action_probability=0.0  # disable sticky actions (v0 behavior)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")

# Build a reverse-action map from action meanings (robust to env variants)
def _dir_vector(name: str):
    name = name.upper()
    dx = (1 if "RIGHT" in name else 0) + (-1 if "LEFT" in name else 0)
    dy = (1 if "DOWN" in name else 0) + (-1 if "UP" in name else 0)
    return (dx, dy)

def build_reverse_map(env):
    meanings = env.unwrapped.get_action_meanings()
    vectors = [_dir_vector(m) for m in meanings]
    reverse = {}
    for i, (dx, dy) in enumerate(vectors):
        opp = (-dx, -dy)
        # find exact opposite if exists
        for j, v in enumerate(vectors):
            if v == opp:
                reverse[i] = j
                break
    return reverse, meanings

REVERSE_MAP, ACTION_MEANINGS = build_reverse_map(env)
NOOP_ACTION = 0 if "NOOP" in ACTION_MEANINGS[0].upper() else 0  # usually 0 is NOOP

# ============================================================
# 2. Preprocessing
# ============================================================
mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2]             # crop and downsize
    img = img.sum(axis=2)               # to greyscale
    img[img == mspacman_color] = 0      # improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize to [-128,127]
    return img.reshape(88, 80, 1)

# ============================================================
# 3. Replay Buffer
# ============================================================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity, batch_size, obs_shape=(4,88,80), device="cuda"):
        self.states = deque([], maxlen=capacity)       # expect np.int8 of shape obs_shape
        self.actions = deque([], maxlen=capacity)      # int (scalar)
        self.rewards = deque([], maxlen=capacity)      # float (scalar)
        self.next_states = deque([], maxlen=capacity)  # np.int8 of shape obs_shape (no None!)
        self.dones = deque([], maxlen=capacity)        # bool
        self.size = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.obs_shape = obs_shape

    def push(self, state_u8, action_int, reward_f, next_state_u8, done_bool):
        # REQUIRE: state_u8 & next_state_u8 are np.ndarray dtype=int8, shape obs_shape
        self.states.append(state_u8)
        self.actions.append(int(action_int))
        self.rewards.append(float(reward_f))
        self.next_states.append(next_state_u8)
        self.dones.append(bool(done_bool))
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        assert self.size >= self.batch_size
        idx = random.sample(range(self.size), k=self.batch_size)

        # Stack safely (no vstack)
        s  = np.stack([self.states[i]      for i in idx], axis=0)   # [B,4,88,80]
        ns = np.stack([self.next_states[i] for i in idx], axis=0)   # [B,4,88,80]
        a  = np.array([self.actions[i]     for i in idx], dtype=np.int64)  # [B]
        r  = np.array([self.rewards[i]     for i in idx], dtype=np.float32)# [B]
        d  = np.array([self.dones[i]       for i in idx], dtype=np.float32)# [B]

        # To torch on device
        s_t  = torch.from_numpy(s).to(self.device, dtype=torch.float32)    # net will /128
        ns_t = torch.from_numpy(ns).to(self.device, dtype=torch.float32)
        a_t  = torch.from_numpy(a).to(self.device).unsqueeze(1)            # [B,1] for gather
        r_t  = torch.from_numpy(r).to(self.device)
        d_t  = torch.from_numpy(d).to(self.device)

        return s_t, a_t, r_t, ns_t, d_t

    def __len__(self):
        return self.size

# ============================================================
# 4. CNN-based Q-network for image input (no BatchNorm; classic DQN)
# ============================================================
class DQN(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        # Expect [B, 4, 88, 80]. We'll center-crop width to 80->80 and height 88->84 path via convs.
        # Classic DQN: conv1(8x8/4), conv2(4x4/2), conv3(3x3/1)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Infer flatten dim dynamically
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
# ============================================================
# 5. Hyperparameters
# ============================================================
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1_000_000  # in steps (frame-based)
TAU = 0.001
LR = 2.5e-4
MEM_CAP = 21_000
WARMUP_STEPS = 20_000   # for v0-like settings; 10k–20k is fine
NUM_EPISODES = 6000
SOLVE_SCORE = 5000  # reference target

# --- control windows & schedules ---
AVOIDED_STEPS = 80     # opening frames to skip controls
DEAD_STEPS    = 36     # steps to skip after life loss
TRAIN_EVERY   = 4      # <-- (Frame skipping for optimization) train every K frames

# Reward shaping config
USE_LOG_REWARD = True
LOG_NORM = 100.0       # larger => gentler scaling
USE_CLIP = True
CLIP_VALUE = 1.0

def transform_reward(r):
    """Signed log-scaling with optional clipping; r is raw game points for this step."""
    if r == 0:
        val = 0.0
    elif USE_LOG_REWARD:
        s = 1.0 if r > 0 else -1.0
        val = s * (math.log1p(abs(r)) / math.log1p(LOG_NORM))
    else:
        val = float(r)
    if USE_CLIP:
        val = max(-CLIP_VALUE, min(CLIP_VALUE, val))
    return val

# ============================================================
# 6. Logging setup
# ============================================================
outdir = Path("results/mspacman_dqn1")
if outdir.exists():
    print(f"[reset] Deleting old directory: {outdir}")
    shutil.rmtree(outdir)
(outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

rewards_csv = outdir / "rewards.csv"
qmax_csv = outdir / "qmax.csv"

with open(rewards_csv, "w", newline="") as f:
    csv.writer(f).writerow(["episode", "return"])
with open(qmax_csv, "w", newline="") as f:
    csv.writer(f).writerow(["episode", "qmax"])

# ============================================================
# 7. Initialization
# ============================================================
n_actions = env.action_space.n
policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEM_CAP, BATCH_SIZE, obs_shape=(4,88,80), device=device.type)


steps_done = 0
last_effective_action = NOOP_ACTION  # to filter reversals on exploration

# ============================================================
# 8. ε-greedy action selection (with reverse-action filter on random branch)
# ============================================================
def select_action(state, eps_threshold):
    global steps_done, last_effective_action
    steps_done += 1
    sample = random.random()

    if sample > eps_threshold:
        with torch.no_grad():
            q = policy_net(state)
            action = q.max(1).indices.view(1, 1)
    else:
        # random exploration — avoid immediate exact reversal if possible
        a = env.action_space.sample()
        rev = REVERSE_MAP.get(last_effective_action, None)
        if rev is not None and a == rev and n_actions > 1:
            # resample from all except the exact reverse
            candidates = [x for x in range(n_actions) if x != rev]
            a = random.choice(candidates)
        action = torch.tensor([[a]], device=device, dtype=torch.long)

    return action

# epsilon schedule tied to steps_done (frame-based)
def current_epsilon():
    frac = math.exp(-1.0 * steps_done / EPS_DECAY)
    return EPS_END + (EPS_START - EPS_END) * frac

# ============================================================
# 9. Training utilities
# ============================================================
def optimize_model():
    if len(memory) < max(BATCH_SIZE, WARMUP_STEPS):
        return

    state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample()

    # Q(s,a)
    q_sa = policy_net(state_batch).gather(1, action_batch).squeeze(1)  # [B]

    # Double DQN target
    with torch.no_grad():
        next_actions = policy_net(next_batch).argmax(1)  # [B]
        next_vals = target_net(next_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # [B]

    target = reward_batch + (1.0 - done_batch) * GAMMA * next_vals

    loss = F.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    
def moving_average(x, k=100):
    if len(x) < 1: return np.array([])
    x = np.asarray(x, dtype=float)
    if len(x) < k: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / float(k)

# ============================================================
# 10. Training loop (with reward-triggered replay + optimization; cooldowns)
# ============================================================
episode_returns, episode_qmax = [], []
seed = 123

for i_episode in tqdm(range(1, NUM_EPISODES + 1)):
    obs, info = env.reset(seed=seed + i_episode)
    prev_lives = info.get("lives", 3)
    cooldown = AVOIDED_STEPS

    total_reward = 0.0
    qmax_ep = -float("inf")

    proc = preprocess_observation(obs)                 # (88, 80, 1)
    frame_stack = deque([proc]*4, maxlen=4)
    stacked = np.concatenate(list(frame_stack), axis=2)  # (88, 80, 4)
    state = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).to(device).float()

    done = False
    while not done:
        # --- skip control during cooldown (start or post-death) ---
        if cooldown > 0:
            # advance env with NOOP to flush frames, but don't store/learn
            next_obs, raw_r, terminated, truncated, info = env.step(NOOP_ACTION)
            done = terminated or truncated
            total_reward += raw_r  # episode return should reflect *raw* score
            # update frame stack
            next_proc = preprocess_observation(next_obs)
            frame_stack.append(next_proc)
            stacked_next = np.concatenate(list(frame_stack), axis=2)
            state = torch.from_numpy(stacked_next).permute(2, 0, 1).unsqueeze(0).to(device).float()
            cooldown -= 1
            if done:
                break
            continue

        # --- ε-greedy action with reverse filter on random branch ---
        eps = current_epsilon()
        action = select_action(state, eps)
        a_int = action.item()

        # --- step env ---
        next_obs, raw_r, terminated, truncated, info = env.step(a_int)
        done = terminated or truncated

        # Track last effective action for reverse filtering
        last_effective_action = a_int

        # Raw episodic return for reporting (human score)
        total_reward += raw_r

        # Transform reward for learning signal
        r = transform_reward(raw_r)

        # --- detect life loss to trigger cooldown and add small penalty/bonus ---
        lives = info.get("lives", prev_lives)
        life_lost = lives < prev_lives
        if life_lost:
            # small learning penalty on life loss
            r += -1.0
            cooldown = DEAD_STEPS
            
        # --- preprocess next state / stack frames (do this BEFORE push) ---
        # --- preprocess next state / stack frames BEFORE push ---
        next_proc = preprocess_observation(next_obs)                       # np.int8 (88,80,1)
        frame_stack.append(next_proc)
        stacked_next = np.concatenate(list(frame_stack), axis=2)           # np.int8 (88,80,4)
        
        # Build replay arrays (np.int8, CHW)
        state_u8      = np.transpose(stacked,      (2,0,1)).astype(np.int8)  # [4,88,80]
        next_state_u8 = np.transpose(stacked_next, (2,0,1)).astype(np.int8)  # [4,88,80]
        
        # Push to replay (NO Nones; plain scalars)
        memory.push(
            state_u8,
            a_int,
            float(r),
            next_state_u8,
            bool(done)
        )
        
        # Prepare next_state for acting (torch float on device)
        next_state = torch.from_numpy(stacked_next).permute(2,0,1).unsqueeze(0).to(device).float()
        
        # optional Q diagnostic
        with torch.no_grad():
            qvals = policy_net(state)
            qmax_ep = max(qmax_ep, float(qvals.max().item()))
        
        # advance
        state = next_state                      # torch float on device
        stacked = stacked_next                  # keep np copy for next push
        prev_lives = lives
        # --- TRAIN-EVERY-K-FRAMES + REWARD-GATED OPTIMIZATION ---
        if steps_done % TRAIN_EVERY == 0:
            optimize_model()

        # --- Polyak update (target smoothing) ---
        with torch.no_grad():
            tgt = target_net.state_dict()
            src = policy_net.state_dict()
            for k in src:
                tgt[k] = src[k] * TAU + tgt[k] * (1 - TAU)
            target_net.load_state_dict(tgt)

    # ---- end of episode bookkeeping ----
    episode_returns.append(total_reward)
    episode_qmax.append(qmax_ep)
    with open(rewards_csv, "a", newline="") as f:
        csv.writer(f).writerow([i_episode, total_reward])
    with open(qmax_csv, "a", newline="") as f:
        csv.writer(f).writerow([i_episode, qmax_ep])

    # Log every 10
    if i_episode % 10 == 0 or i_episode == 1:
        ma100 = moving_average(episode_returns, 100)
        ma_val = ma100[-1] if len(ma100) else float("nan")
        print(f"[ep {i_episode:5d}] return={total_reward:6.1f}   MA100={ma_val:6.1f}   eps={current_epsilon():5.3f}   buffer={len(memory)}")

    # Checkpoint every 100
    if i_episode % 100 == 0:
        ckpt_path = outdir / "checkpoints" / f"ckpt_ep{i_episode}.pt"
        torch.save(policy_net.state_dict(), ckpt_path)
        print(f"[ckpt] saved {ckpt_path}")