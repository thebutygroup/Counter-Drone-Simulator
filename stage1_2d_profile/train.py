"""
train.py  (stage 1: 2D profile view)
====================================
Headless training with checkpointing, resume, and a metrics log. Everything is
written into THIS folder so `server.py --agent` finds the policy.

Artifacts produced here:
    q_policy.npz        deployable policy (server.py --agent loads this)
    checkpoint.npz      full training state (resumable; crash-safe)
    metrics.csv         append-only progress log (open in Excel)
    training_curve.png  success/reward curve

Run:
    python train.py                          # fresh curriculum, 12000 episodes
    python train.py --episodes 6000
    python train.py --resume --episodes 4000 # continue from checkpoint.npz (EVADE regime)
"""

import argparse
import os
import sys
import matplotlib
import numpy as np

# --- make the shared engine in ../core importable, regardless of cwd ---------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from qlearn import (run_curriculum, continue_training, GreedyAgent,
                    plot_history, evaluate)

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(HERE, "q_policy.npz")
CKPT = os.path.join(HERE, "checkpoint.npz")
METRICS = os.path.join(HERE, "metrics.csv")
CURVE = os.path.join(HERE, "training_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoint.npz instead of starting fresh")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="write checkpoint + metrics row every N episodes")
    args = ap.parse_args()

    np.random.seed(args.seed)

    if args.resume:
        if not os.path.exists(CKPT):
            sys.exit(f"No checkpoint found at {CKPT} -- run without --resume first.")
        Q, disc, history, split = continue_training(
            CKPT, args.episodes, seed=args.seed,
            ckpt_every=args.checkpoint_every, metrics_csv=METRICS)
    else:
        Q, disc, history, split = run_curriculum(
            args.episodes, seed=args.seed,
            ckpt_path=CKPT, ckpt_every=args.checkpoint_every, metrics_csv=METRICS)

    # Final saves: deployable policy + curve (checkpoint already written by the loop).
    agent = GreedyAgent(Q, disc)
    agent.save(POLICY)
    plot_history(history, split, path=CURVE)
    print(f"\nSaved:\n  policy     -> {POLICY}\n  checkpoint -> {CKPT}"
          f"\n  metrics    -> {METRICS}\n  curve      -> {CURVE}")

    print("--- eval vs EVADING ---");     evaluate(agent, evasion=True)
    print("--- eval vs NON-evading ---"); evaluate(agent, evasion=False)


if __name__ == "__main__":
    main()
