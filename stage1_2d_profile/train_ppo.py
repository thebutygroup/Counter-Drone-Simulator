"""
train_ppo.py  (stage 1: 2D profile view)  --  PPO replacement for DQN
=====================================================================
Proximal Policy Optimization (Stable-Baselines3) on the SAME 2D env, same
curriculum, and same final eval as train_dqn.py -- so the printed success/crash/
impact lines are directly comparable to a DQN run.
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
import drone_core
from drone_core import SimulationConfig # Import the new config class

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "ppo_policy")          # .zip appended by SB3
CKPT_DIR = os.path.join(HERE, "ppo_checkpoints")

# --- hyperparameters (tune these) -------------------------------------------
HYPERPARAMS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)

# curriculum: (label, fraction_of_budget, evasion, closing_scale, target_frac)
CURRICULUM = [
    ("survive",     0.15, False, 0.00, 0.35),
    ("catch",       0.20, False, 0.10, 0.45),
    ("evade-slow",  0.20, True,  0.10, 0.45),
    ("evade-mid",   0.20, True,  0.10, 0.60),
    ("evade-full",  0.25, True,  0.10, 0.75),
]


# Swapped `target_speed` for `config`
def make_vec(evasion, closing, n_envs, config, seed=None, n_obstacles=0,
             perception="none", perception_kwargs=None, continuous=False):
    """A vectorised stack of DroneGymEnvs."""
    return make_vec_env(
        DroneGymEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(evasion=evasion, closing_scale=closing, config=config,
                        n_obstacles=n_obstacles, perception=perception,
                        perception_kwargs=perception_kwargs or {}, continuous=continuous),
        vec_env_cls=DummyVecEnv,
    )


# Swapped `target_speed` for `config`
def evaluate(model, config, episodes=300, evasion=True, seed=999,
             n_obstacles=0, perception="none", perception_kwargs=None, continuous=False):
    """Single-env, deterministic eval."""
    env = DroneGymEnv(evasion=evasion, closing_scale=0.10, seed=seed,
                      config=config, n_obstacles=n_obstacles,
                      perception=perception, perception_kwargs=perception_kwargs or {},
                      continuous=continuous)
    wins = crashes = escapes = 0
    steps, impacts, backs = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        term = trunc = False
        info = {}
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            act = action if continuous else int(action)
            obs, _, term, trunc, info = env.step(act)
        if info.get("success"):
            wins += 1; steps.append(info["steps"]); impacts.append(info["impact_speed"])
            backs.append(info.get("backside", 0.0))
        elif info.get("escaped"):
            escapes += 1
        elif info.get("crashed"):
            crashes += 1
    sr, cr, er = wins / episodes, crashes / episodes, escapes / episodes
    avg = np.mean(steps) if steps else float("nan")
    spd = np.mean(impacts) if impacts else float("nan")
    bck = np.mean(backs) if backs else float("nan")
    print(f"Eval: success={sr:.1%}  crash={cr:.1%}  escape={er:.1%} over {episodes} eps "
          f"(avg catch {avg:.0f} frames, impact {spd:.1f} px/frame, backside {bck:.2f})")
    return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from ppo_policy.zip")
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--perception", choices=["none", "rays", "slots"], default="none")
    ap.add_argument("--rays", type=int, default=16)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--target-scale", type=float, default=1.0)
    ap.add_argument("--target-frac", type=float, default=0.75)
    ap.add_argument("--continuous", action="store_true")
    args = ap.parse_args()

    pkw = {}
    if args.perception == "rays":
        pkw = {"n_rays": args.rays}
    elif args.perception == "slots":
        pkw = {"k": args.slots}

    os.makedirs(CKPT_DIR, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=CKPT_DIR, name_prefix="ppo")

    phases = ([("evade", 1.0, True, 0.10, args.target_frac)]
              if args.no_curriculum else CURRICULUM)

    # Build first config and env
    first = phases[0]
    first_config = SimulationConfig(target_speed_frac=first[4] * args.target_scale)
    
    venv0 = make_vec(evasion=first[2], closing=first[3], n_envs=args.n_envs, seed=args.seed,
                     config=first_config, n_obstacles=args.obstacles,
                     perception=args.perception, perception_kwargs=pkw,
                     continuous=args.continuous)
                     
    if args.resume and os.path.exists(MODEL_PATH + ".zip"):
        print(f"Resuming from {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=venv0)
    else:
        model = PPO(env=venv0, seed=args.seed, **HYPERPARAMS)

    # Run the curriculum
    for i, (label, frac, evasion, closing, tfrac) in enumerate(phases):
        steps = int(args.timesteps * frac)
        
        # Create a specific config for this curriculum phase!
        phase_config = SimulationConfig(target_speed_frac=tfrac * args.target_scale)
        ts = phase_config.target_stats['target_speed'] # just for the print log
        
        print(f"\n=== Phase '{label}': {steps} steps "
              f"(evasion={evasion}, closing={closing}, target={ts:.1f}px/f "
              f"[{tfrac:.2f}x top], n_envs={args.n_envs}, "
              f"obstacles={args.obstacles}, perception={args.perception}) ===")
              
        if i > 0:
            model.set_env(make_vec(evasion=evasion, closing=closing, n_envs=args.n_envs,
                                   seed=args.seed + i, config=phase_config,
                                   n_obstacles=args.obstacles,
                                   perception=args.perception, perception_kwargs=pkw,
                                   continuous=args.continuous))
                                   
        model.learn(total_timesteps=steps, callback=checkpoint_cb,
                    reset_num_timesteps=(i == 0 and not args.resume),
                    log_interval=20)
        model.save(MODEL_PATH)
        print(f"saved -> {MODEL_PATH}.zip")

    # Eval at the FULL final difficulty
    final_tfrac = phases[-1][4]
    final_config = SimulationConfig(target_speed_frac=final_tfrac * args.target_scale)
    full_ts = final_config.target_stats['target_speed']
    
    print(f"\n--- final eval (target {full_ts:.1f}px/f = {final_tfrac*args.target_scale:.2f}x top) ---")
    evaluate(model, config=final_config, evasion=True, n_obstacles=args.obstacles,
             perception=args.perception, perception_kwargs=pkw, continuous=args.continuous)
    evaluate(model, config=final_config, evasion=False, n_obstacles=args.obstacles,
             perception=args.perception, perception_kwargs=pkw, continuous=args.continuous)


if __name__ == "__main__":
    main()