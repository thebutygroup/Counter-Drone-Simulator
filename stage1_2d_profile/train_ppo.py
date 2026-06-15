"""
train_ppo.py  (stage 1: 2D profile view)  --  PPO replacement for DQN
=====================================================================
Proximal Policy Optimization (Stable-Baselines3) on the SAME 2D env, same
curriculum, and same final eval as train_dqn.py -- so the printed success/crash/
impact lines are directly comparable to a DQN run.

WHY PPO: it is the go-forward algorithm for the project. The hard "fast +
backside-first" reward (A3 gate) wants on-policy policy-gradient timing, and the
self-play / multi-agent steps (B/C) are built on PPO regardless. Switching now
means the additive reward (A1+A2+A3) is tuned once on the algorithm we keep.

INSTALL (same extras as DQN):
    pip install "stable-baselines3[extra]"

RUN:
    python train_ppo.py                       # curriculum, ~1.5M steps, 8 envs
    python train_ppo.py --timesteps 3000000   # PPO often wants MORE steps than DQN
    python train_ppo.py --no-curriculum       # single-phase (evade only)
    python train_ppo.py --resume              # continue from ppo_policy.zip
    python train_ppo.py --n-envs 4            # fewer parallel envs (less RAM/CPU)

OUTPUTS (in this folder):
    ppo_policy.zip          final model (load with PPO.load)
    ppo_checkpoints/        periodic checkpoints (crash-safe / resumable)

NOTE: written without torch in-house (not executed here) -- treat the first run
as a smoke test and report any SB3/gymnasium tracebacks.
"""

import argparse
import os
import sys

import numpy as np

# --- make ../core importable ------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_env import DroneGymEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "ppo_policy")          # .zip appended by SB3
CKPT_DIR = os.path.join(HERE, "ppo_checkpoints")

# --- hyperparameters (tune these) -------------------------------------------
# On-policy: no replay buffer / learning_starts / target net / epsilon. PPO
# explores via a STOCHASTIC policy + an entropy bonus, so ent_coef matters early
# (this task needs exploration to stumble into intercepts).
HYPERPARAMS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[256, 256]),  # match DQN capacity for a fair compare
    learning_rate=3e-4,
    n_steps=2048,            # rollout length PER ENV (buffer = n_steps * n_envs)
    batch_size=512,          # must divide n_steps * n_envs
    n_epochs=10,             # optimisation passes per rollout
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,           # entropy bonus -> exploration; lower it if it won't commit
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)

# curriculum: (label, fraction_of_budget, evasion, closing_scale) -- IDENTICAL to DQN
CURRICULUM = [
    ("survive", 0.25, False, 0.0),
    ("catch",   0.35, False, 0.10),
    ("evade",   0.40, True,  0.10),
]


def make_vec(evasion, closing, n_envs, seed=None):
    """A vectorised stack of DroneGymEnvs (Monitor-wrapped by make_vec_env).
    Each sub-env is seeded distinctly so the parallel rollouts aren't identical."""
    return make_vec_env(
        DroneGymEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(evasion=evasion, closing_scale=closing),
        vec_env_cls=DummyVecEnv,   # in-process; robust on Windows. Swap to Subproc for speed.
    )


def evaluate(model, episodes=300, evasion=True, seed=999):
    """Single-env, deterministic eval -- SAME harness as train_dqn.evaluate so the
    numbers line up directly with the DQN run."""
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
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from ppo_policy.zip")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    # save_freq is counted in MODEL steps; divide by n_envs so checkpoints land at
    # roughly the intended total-timestep cadence regardless of env count.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=CKPT_DIR, name_prefix="ppo")

    phases = ([("evade", 1.0, True, 0.10)] if args.no_curriculum else CURRICULUM)

    # Build or load the model on the first phase's vec env.
    first = phases[0]
    venv0 = make_vec(evasion=first[2], closing=first[3], n_envs=args.n_envs, seed=args.seed)
    if args.resume and os.path.exists(MODEL_PATH + ".zip"):
        print(f"Resuming from {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=venv0)
    else:
        model = PPO(env=venv0, seed=args.seed, **HYPERPARAMS)

    # Run the curriculum: keep the same model, swap the vec env per phase.
    for i, (label, frac, evasion, closing) in enumerate(phases):
        steps = int(args.timesteps * frac)
        print(f"\n=== Phase '{label}': {steps} steps "
              f"(evasion={evasion}, closing={closing}, n_envs={args.n_envs}) ===")
        if i > 0:
            model.set_env(make_vec(evasion=evasion, closing=closing,
                                   n_envs=args.n_envs, seed=args.seed + i))
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
