"""
train.py  (stage 1: 2D profile view)
====================================
Headless training (no browser). Runs the 3-phase curriculum, saves the policy
and a learning curve into THIS folder so `server.py --agent` finds them.

Run:
    python train.py                  # default 12000 episodes
    python train.py --episodes 6000
"""

import argparse
import os
import sys

import numpy as np

# --- make the shared engine in ../core importable, regardless of cwd ---------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from qlearn import run_curriculum, GreedyAgent, plot_history, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    Q, disc, history, split = run_curriculum(args.episodes, seed=args.seed)

    GreedyAgent(Q, disc).save(os.path.join(HERE, "q_policy.npz"))
    plot_history(history, split, path=os.path.join(HERE, "training_curve.png"))
    print(f"Saved policy + curve into {HERE}")

    print("--- eval vs EVADING ---");     evaluate(GreedyAgent(Q, disc), evasion=True)
    print("--- eval vs NON-evading ---"); evaluate(GreedyAgent(Q, disc), evasion=False)


if __name__ == "__main__":
    main()
