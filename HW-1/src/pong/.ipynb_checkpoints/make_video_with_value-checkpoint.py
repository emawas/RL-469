# src/pong/make_video_with_value.py
import argparse, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py

from PIL import Image, ImageDraw, ImageFont

from src.pong.preprocess import preprocess   # (H,W,3)->(80,80) float32 [0,1]
from src.pong.policy_cnn import PolicyCNN, select_action_greedy

# ---- Value network for Pong (mirror of PolicyCNN conv stack, scalar head) ----
class ValueCNN(nn.Module):
    def __init__(self, hidden: int = 200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 80, 80)
            flat = self.conv(dummy).view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        z = self.conv(x)
        v = self.head(z)
        return v.squeeze(-1)  # (B,)

def pick_device(name: str):
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def find_latest_checkpoint(run_dir: Path | None, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if run_dir is None:
        raise FileNotFoundError("Provide --run_dir or --checkpoint")
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        pts = sorted(ckpt_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt files in {ckpt_dir}")
        pat = re.compile(r"ep(\d+)\.pt$")
        with_eps = []
        for p in pts:
            m = pat.search(p.name)
            if m: with_eps.append((int(m.group(1)), p))
        if with_eps:
            with_eps.sort(key=lambda t: t[0])
            return with_eps[-1][1]
        pts.sort(key=lambda p: p.stat().st_mtime)
        return pts[-1]
    # fallbacks
    for name in ("checkpoint.pt", "policy.pt"):
        p = run_dir / name
        if p.exists(): return p
    raise FileNotFoundError(f"No checkpoints found under {run_dir}")

def load_policy_and_value(policy: PolicyCNN, value: ValueCNN, path: Path, device):
    blob = torch.load(str(path), map_location=device)
    if isinstance(blob, dict) and "policy" in blob:
        policy.load_state_dict(blob["policy"])
        if "value" in blob:
            value.load_state_dict(blob["value"])
        else:
            # try to load whole dict into value (if user saved separately)
            try:
                value.load_state_dict(blob)
            except Exception:
                print("[video] WARNING: no 'value' weights in checkpoint; value overlay will be zeros.")
                value.apply(lambda m: None)  # noop
    else:
        # maybe it's a plain state_dict of the policy only
        policy.load_state_dict(blob)
        print("[video] WARNING: checkpoint looked like policy-only; value overlay will be zeros.")

def put_overlay(img_rgb, text: str):
    # img_rgb: np.uint8 (H,W,3)
    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im)
    try:
        # will default to a basic font if not found
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    # light outline
    x, y = 8, 8
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), text, fill=(0,0,0), font=font)
    draw.text((x, y), text, fill=(255,255,255), font=font)
    return np.array(im)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="", help="Run dir containing checkpoints/")
    ap.add_argument("--checkpoint", type=str, default="", help="Explicit checkpoint .pt")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="", help="Output path (.mp4 or .gif)")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=200, help="Hidden size used in training")
    ap.add_argument("--frame_diff", action="store_true", help="Use frame differencing like training")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[video] device: {device}")

    run_dir = Path(args.run_dir) if args.run_dir else None
    ckpt = find_latest_checkpoint(run_dir, Path(args.checkpoint) if args.checkpoint else None)
    print(f"[video] using checkpoint: {ckpt}")

    out_path = Path(args.out) if args.out else ( (run_dir or ckpt.parent) / "episode_with_value.mp4" )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[video] writing: {out_path}")

    # Env
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)
    # FIRE to serve
    obs, _, _, _, _ = env.step(1)

    # Models
    policy = PolicyCNN(hidden=args.hidden).to(device).eval()
    valuef = ValueCNN(hidden=args.hidden).to(device).eval()
    load_policy_and_value(policy, valuef, ckpt, device)

    # Video writer
    writer = None
    frames = None
    try:
        import imageio.v3 as iio
        if out_path.suffix.lower() == ".mp4":
            writer = iio.get_writer(out_path, fps=args.fps, codec="libx264")
        else:
            writer = iio.get_writer(out_path, fps=args.fps)
    except Exception:
        import imageio
        print("[video] imageio.v3/libx264 unavailable; will buffer frames and save GIF.")
        frames = []
        writer = None
        imageio_fallback = imageio

    done = False
    total_r = 0.0
    steps = 0
    prev_proc = None

    while not done:
        rgb = obs  # numpy uint8 (H,W,3)

        # preprocess frame
        proc = preprocess(rgb)  # (80,80) float32
        if args.frame_diff and prev_proc is not None:
            frame_in = proc - prev_proc
        else:
            frame_in = proc
        prev_proc = proc

        # action (greedy)
        a = select_action_greedy(policy, frame_in, device=device)

        # value estimate
        x = torch.as_tensor(frame_in, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            v = float(valuef(x).item())

        # overlay HUD
        hud = f"step: {steps}   V(s): {v:+.3f}   return: {total_r:+.1f}"
        rgb_hud = put_overlay(rgb, hud)

        if writer is not None:
            writer.append_data(rgb_hud)
        else:
            frames.append(rgb_hud)

        obs, r, term, trunc, _ = env.step(a)
        total_r += float(r)
        steps += 1
        if term or trunc or steps > 10000:
            break

    env.close()

    if writer is not None:
        writer.close()
    else:
        if out_path.suffix.lower() != ".gif":
            out_path = out_path.with_suffix(".gif")
        imageio_fallback.mimsave(out_path, frames, fps=args.fps)

    print(f"[video] done: {out_path} | steps={steps} | return={total_r:+.1f}")

if __name__ == "__main__":
    main()