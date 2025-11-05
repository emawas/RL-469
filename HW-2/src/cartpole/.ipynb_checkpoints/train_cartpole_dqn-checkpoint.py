# src/cartpole/train_cartpole_dqn.py
import argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque, namedtuple
import random
import time
import shutil

# ------------------------------
# Utils & Replay Buffer
# ------------------------------
Transition = namedtuple("Transition", "s a r s2 done")

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        device: torch.device,
        alpha: float = 0.6,
        use_per: bool = True,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.use_per = use_per           # <— toggle on/off
        self.eps_prio = 1e-6
        self.idx = 0
        self.full = False

        # data
        self.S    = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.A    = np.zeros((capacity, 1),       dtype=np.int64)
        self.R    = np.zeros((capacity, 1),       dtype=np.float32)
        self.S2   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.Done = np.zeros((capacity, 1),       dtype=np.float32)

        # priorities (only used if use_per=True)
        self.prio = np.ones((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s2, done):
        i = self.idx
        self.S[i] = s
        self.A[i, 0] = a
        self.R[i, 0] = r
        self.S2[i] = s2
        self.Done[i, 0] = done

        # if PER on, new sample gets high priority
        if self.use_per:
            max_prio = self.prio.max() if (self.full or self.idx > 0) else 1.0
            self.prio[i, 0] = max_prio

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self)

        if self.use_per:
            # --- PER sampling ---
            prios = self.prio[:n, 0]
            probs = prios ** self.alpha
            probs /= probs.sum()
            idxs = np.random.choice(n, size=batch_size, p=probs)

            weights = (n * probs[idxs]) ** (-beta)
            weights /= weights.max()
        else:
            # --- Uniform sampling ---
            idxs = np.random.randint(0, n, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)

        # fetch batch tensors
        S    = torch.as_tensor(self.S[idxs],    device=self.device)
        A    = torch.as_tensor(self.A[idxs],    device=self.device)
        R    = torch.as_tensor(self.R[idxs],    device=self.device)
        S2   = torch.as_tensor(self.S2[idxs],   device=self.device)
        Done = torch.as_tensor(self.Done[idxs], device=self.device)
        W    = torch.as_tensor(weights.reshape(-1,1), dtype=torch.float32, device=self.device)
        return S, A, R, S2, Done, W, idxs

    def update_priorities(self, idxs, td_errors):
        # only update if PER is active
        if not self.use_per:
            return
        # after computing td
        td = np.abs(td_errors).reshape(-1)
        td = np.minimum(td, 10.0)          # clip huge spikes
        self.prio[idxs, 0] = td + self.eps_prio
# ------------------------------
# Q-Network (MLP)
# ------------------------------
class QNet(nn.Module):
    """Qθ(s,·): outputs a vector of Q-values for each discrete action."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, obs_dim) -> returns (B, act_dim) of Q-values
        return self.net(x)


# ------------------------------
# ε-greedy policy
# ------------------------------
def select_action(qnet: QNet, state, eps: float, act_dim: int, device: torch.device):
    # DQN is an OFF-POLICY value method; the "policy" during training is ε-greedy on Q.
    if random.random() < eps:
        return random.randrange(act_dim)  # explore
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1,obs)
        q = qnet(s)                         # (1, act_dim)
        a = torch.argmax(q, dim=1).item()   # greedy action from online Q
        return a


# ------------------------------
# Target network update helpers
# ------------------------------
def hard_update(target: nn.Module, online: nn.Module):
    target.load_state_dict(online.state_dict())  # <-- DQN target network: sync params


def polyak_update(target: nn.Module, online: nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * op.data)


# ------------------------------
# Moving average for logging
# ------------------------------
def moving_average(x, k=100):
    if len(x) < 1: return np.array([])
    x = np.asarray(x, dtype=float)
    if len(x) < k: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / float(k)


# ------------------------------
# Train DQN
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.95)          # <-- assignment requirement
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--buffer_size", type=int, default=50000)     # experience replay capacity
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=20000) # linear decay over steps
    ap.add_argument("--target_update_every", type=int, default=1000) # hard update frequency (steps)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_polyak", action="store_true", help="Use Polyak updates instead of hard")
    ap.add_argument("--tau", type=float, default=0.005, help="Polyak factor if --use_polyak")
    ap.add_argument("--results_dir", type=str, default="results/cartpole_dqn")
    ap.add_argument("--overwrite", action="store_true", help="If set, delete results_dir before starting.")
    ap.add_argument("--use_per", action="store_true", help="Enable prioritized replay buffer")
    
    args = ap.parse_args()
    outdir = Path(args.results_dir)
    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    # Seeding
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Env
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=args.seed)
    obs_dim = obs.shape[0]
    act_dim = env.action_space.n

    # DQN components
    q_online = QNet(obs_dim, act_dim, hidden=args.hidden).to(device)   # online network Qθ
    q_target = QNet(obs_dim, act_dim, hidden=args.hidden).to(device)   # target network Qθ^-
    hard_update(q_target, q_online)   # <-- DQN: initialize target = online

    optimizer = optim.Adam(q_online.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()       # Huber loss (robust) — common for DQN

    # Experience Replay
    alpha, beta_start, beta_end, beta_steps = 0.6, 0.4, 1.0, 200000
    replay = ReplayBuffer(args.buffer_size, obs_dim, device, alpha=0.6, use_per=args.use_per)
    beta = beta_start
    beta_inc = (beta_end - beta_start) / max(1, beta_steps)

    # Logging
    outdir = Path(args.results_dir); outdir.mkdir(parents=True, exist_ok=True)
    rewards_csv = outdir / "rewards.csv"
    with open(rewards_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return"])

    # ε schedule (linear)
    eps = args.eps_start
    eps_slope = (args.eps_end - args.eps_start) / float(max(1, args.eps_decay_steps))

    all_returns = []
    global_step = 0

    
    # one-time at setup:
    qmax_csv = outdir / "qmax.csv"
    with open(qmax_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "qmax"])
    for ep in range(1, args.episodes + 1):
        qmax_ep = -float("inf")  # <-- reset each episode
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0

        while not done:
            # ---- ε-greedy action from Q (DQN policy during training) ----
            a = select_action(q_online, obs, eps, act_dim, device)

            # Interact with env
            next_obs, r, terminated, truncated, info = env.step(a)
            
            # Bookkeeping
            ep_ret += r
            
            # --- Termination semantics ---
            true_done    = bool(terminated)                                # pole fell (real terminal)
            timeout      = bool(truncated) and ("TimeLimit.truncated" in info)  # time-limit end
            episode_done = bool(terminated or truncated)                   # for loop/control flow only
            
            # Store transition for learning:
            # Use ONLY true terminal to cut bootstrapping; timeouts keep bootstrap.
            replay.push(obs, a, r, next_obs, float(true_done))
            
            # Step counters / qmax logging
            global_step += 1
            with torch.no_grad():
                s_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                qmax_ep = max(qmax_ep, float(q_online(s_t).max().item()))
            
            # Advance state
            obs = next_obs
            
            # End episode cleanly on either failure or timeout
            if episode_done:
                break
            # Linearly decay ε
            if eps > args.eps_end:
                eps = max(args.eps_end, eps + eps_slope)

            # Learn if we have enough samples
            if len(replay) >= max(args.batch_size,1000):
                for _ in range(2):
                    # sample with priorities + IS weights
                    S, A, R, S2, Done, W, idxs = replay.sample(args.batch_size, beta=beta)
            
                    with torch.no_grad():
                        # Use online net to select the next action
                        next_actions = q_online(S2).argmax(dim=1, keepdim=True)
                    
                        # Use target net to evaluate that action
                        q_next = q_target(S2).gather(1, next_actions)
                    
                        # Compute Double DQN target
                        y = R + (1.0 - Done) * args.gamma * q_next
            
                    q_sa = q_online(S).gather(1, A)
                    td_errors = (y - q_sa).detach().cpu().numpy()
            
                    # weighted Huber loss
                    elementwise = torch.nn.functional.smooth_l1_loss(q_sa, y, reduction="none")
                    loss = (W * elementwise).mean()
            
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_online.parameters(), 5.0)
                    optimizer.step()
            
                    # update priorities using |TD error|
                    replay.update_priorities(idxs, td_errors)
            
                # advance beta slowly toward 1.0
                beta = min(beta_end, beta + beta_inc)
                # ------------------------------
                # Target network update (stabilize bootstrapping)
                # ------------------------------
                if args.use_polyak:
                    polyak_update(q_target, q_online, tau=args.tau)  # soft update
                else:
                    if global_step % args.target_update_every == 0:
                        hard_update(q_target, q_online)              # hard update

        # Logging per episode
        all_returns.append(ep_ret)
        with open(rewards_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, ep_ret])
        # per-episode:
        with open(qmax_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, qmax_ep])
            
        if ep % 50 == 0 or ep == 1:
            ma100 = moving_average(all_returns, 100)
            ma_val = ma100[-1] if len(ma100) else float("nan")
            print(f"[ep {ep:5d}] return={ep_ret:6.1f}   MA100={ma_val:6.1f}   eps={eps:5.3f}   buffer={len(replay)}")

        # compute MA100 each episode
        ma100 = moving_average(all_returns, 100)
        if len(ma100) and ma100[-1] >= 500:
            print(f"[solve] MA100 reached 500 at episode {ep}. Stopping.")
            break

        # ---- Save checkpoint every 100 episodes ----
        if ep % 100 == 0:
            ckpt_dir = outdir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"ckpt_ep{ep}.pt"
            torch.save(
                {
                    "q_online": q_online.state_dict(),
                    "q_target": q_target.state_dict(),
                    "config": vars(args),
                    "episode": ep,
                    "return": ep_ret,
                    "qmax_ep": qmax_ep,
                },
                ckpt_path
            )
            print(f"[ckpt] Saved checkpoint at episode {ep}: {ckpt_path}")

    # ---- Final save after training loop ----
    torch.save(
        {
            "q_online": q_online.state_dict(),
            "q_target": q_target.state_dict(),
            "config": vars(args),
            "episodes_trained": len(all_returns),
            "ma100_final": (moving_average(all_returns, 100)[-1] if len(all_returns) >= 100 else float('nan')),
        },
        outdir / "dqn_checkpoint.pt"
    )
    print(f"[done] saved to {outdir}")

if __name__ == "__main__":
    main() 