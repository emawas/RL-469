#!/usr/bin/env python3
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ma(x, k=100):
    x = np.asarray(x, float)
    if x.size < 1: return x
    if x.size < k: return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / float(k)

def load_csv_safely(p):
    try:
        if p.exists():
            df = pd.read_csv(p)
            # keep only numeric, drop NaNs
            return df.select_dtypes(include=[np.number]).dropna()
    except Exception:
        pass
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="Path containing rewards.csv and (optionally) qmax.csv")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="Seconds between refreshes")
    ap.add_argument("--show", action="store_true",
                    help="Show an interactive window as well")
    ap.add_argument("--outfile", type=str, default="live_learning_curve.png",
                    help="Output PNG filename written under run_dir")
    args = ap.parse_args()

    run = Path(args.run_dir)
    rewards_csv = run / "rewards.csv"
    qmax_csv    = run / "qmax.csv"
    out_png     = run / args.outfile

    # Headless-safe defaults
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), dpi=140)
    fig.tight_layout(pad=2.0)

    if args.show:
        plt.ion()
        plt.show(block=False)

    last_rewards_rows = 0
    last_qmax_rows = 0

    while True:
        dfR = load_csv_safely(rewards_csv)   # expects columns: episode, return
        dfQ = load_csv_safely(qmax_csv)      # expects columns: episode, qmax

        # Plot (ii): episode rewards + MA(100)
        ax1.clear()
        ax1.set_title("Episode Return (with MA100)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Return")

        if not dfR.empty and "return" in dfR.columns:
            ep = np.arange(1, len(dfR["return"]) + 1)
            ret = dfR["return"].to_numpy()
            ax1.plot(ep, ret, linewidth=1.2, label="Return")
            ma100 = ma(ret, 100)
            if ma100.size:
                ax1.plot(np.arange(len(ret) - len(ma100) + 1, len(ret) + 1),
                         ma100, linewidth=2.0, label="MA(100)")
            ax1.legend(loc="upper left")

        # Plot (i): max Q per episode
        ax2.clear()
        ax2.set_title("Max Q per Episode")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("max Q(s, a)")
        if not dfQ.empty and "qmax" in dfQ.columns:
            epQ = np.arange(1, len(dfQ["qmax"]) + 1)
            qv  = dfQ["qmax"].to_numpy()
            ax2.plot(epQ, qv, linewidth=1.2, label="max Q")
            ax2.legend(loc="upper left")

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

        fig.savefig(out_png, bbox_inches="tight")

        # If not showing interactively, just keep saving & sleeping
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()