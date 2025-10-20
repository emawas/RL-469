import argparse
import torch
import gymnasium as gym
from pathlib import Path
from src.cartpole.policy_mlp import Policy
from src.common.rollouts import select_action_greedy  # reuse the greedy above

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="results/cartpole_demo")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Gymnasium video recorder needs render_mode="rgb_array" and wrapper
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=str(outdir), name_prefix="cartpole_demo")

    obs, info = env.reset(seed=args.seed)
    obs_dim = len(obs); act_dim = env.action_space.n

    policy = Policy(obs_dim, act_dim, hidden=128).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    done, ep_ret = False, 0.0
    while not done:
        a = select_action_greedy(policy, obs, device=device)
        obs, r, terminated, truncated, _ = env.step(a)
        ep_ret += float(r)
        done = terminated or truncated

    env.close()
    print(f"[video] saved under: {outdir} (return={ep_ret:.1f})")

if __name__ == "__main__":
    main()