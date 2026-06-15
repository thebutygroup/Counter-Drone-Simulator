"""
train_dqn.py  (stage 1: 2D profile view)  --  NEURAL replacement for qlearn
===========================================================================
Deep Q-Network (Stable-Baselines3) on the SAME 2D env the tabular agent used.
This is the experiment to see whether a neural value function breaks the tabular
plateau before we move to PyBullet/3D.

INSTALL (on your machine; not pre-installed):
    pip install "stable-baselines3[extra]"
    # pulls in torch, gymnasium, tensorboard

RUN:
    python train_dqn.py                      # curriculum, ~1.5M steps
    python train_dqn.py --timesteps 600000   # shorter
    python train_dqn.py --no-curriculum      # single-phase (evade only)

OUTPUTS (in this folder):
    dqn_policy.zip          final model (load with DQN.load)
    dqn_checkpoints/        periodic checkpoints (crash-safe / resumable)

NOTE: This file was written without being executed in-house (no torch here), so
treat the first run as a smoke test and report any SB3/gymnasium API errors.
"""

import argparse
import os
import sys

import numpy as np

# --- make ../core importable ------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gym_env import DroneGymEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "dqn_policy")          # .zip appended by SB3
CKPT_DIR = os.path.join(HERE, "dqn_checkpoints")

# --- hyperparameters (tune these) -------------------------------------------
HYPERPARAMS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=200_000,
    learning_starts=5_000,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=2_000,
    exploration_fraction=0.3,        # fraction of EACH phase spent annealing eps
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
)

# curriculum: (label, fraction_of_budget, evasion, closing_scale)
CURRICULUM = [
    ("survive", 0.25, False, 0.0),
    ("catch",   0.35, False, 0.10),
    ("evade",   0.40, True,  0.10),
]


def make_env(evasion, closing, seed=None):
    return Monitor(DroneGymEnv(evasion=evasion, closing_scale=closing, seed=seed))


def evaluate(model, episodes=300, evasion=True, seed=999):
    env = DroneGymEnv(evasion=evasion, closing_scale=0.10, seed=seed)
    wins = crashes = 0
    steps, impacts = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        term = trunc = False
        info = {}
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
        if info.get("success"):
            wins += 1; steps.append(info["steps"]); impacts.append(info["impact_speed"])
        elif info.get("crashed"):
            crashes += 1
    sr, cr = wins / episodes, crashes / episodes
    avg = np.mean(steps) if steps else float("nan")
    spd = np.mean(impacts) if impacts else float("nan")
    print(f"Eval: success={sr:.1%}  crash={cr:.1%} over {episodes} eps "
          f"(avg catch {avg:.0f} frames, impact {spd:.1f} px/frame)")
    return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from dqn_policy.zip")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000, save_path=CKPT_DIR, name_prefix="dqn")

    phases = ([("evade", 1.0, True, 0.10)] if args.no_curriculum else CURRICULUM)

    # Build or load the model on the first phase's env.
    first = phases[0]
    env0 = make_env(evasion=first[2], closing=first[3], seed=args.seed)
    if args.resume and os.path.exists(MODEL_PATH + ".zip"):
        print(f"Resuming from {MODEL_PATH}.zip")
        model = DQN.load(MODEL_PATH, env=env0)
    else:
        model = DQN(env=env0, seed=args.seed, **HYPERPARAMS)

    # Run the curriculum: keep the same model, swap the env per phase.
    for i, (label, frac, evasion, closing) in enumerate(phases):
        steps = int(args.timesteps * frac)
        print(f"\n=== Phase '{label}': {steps} steps "
              f"(evasion={evasion}, closing={closing}) ===")
        if i > 0:
            model.set_env(make_env(evasion=evasion, closing=closing, seed=args.seed + i))
        model.learn(total_timesteps=steps, callback=checkpoint_cb,
                    reset_num_timesteps=(i == 0 and not args.resume),
                    log_interval=20)
        model.save(MODEL_PATH)
        print(f"saved -> {MODEL_PATH}.zip")

    print("\n--- final eval ---")
    evaluate(model, evasion=True)
    evaluate(model, evasion=False)


if __name__ == "__main__":
    main()
