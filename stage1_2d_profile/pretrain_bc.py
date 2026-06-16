"""
pretrain_bc.py  (stage 1)
=========================
Behaviour-cloning warm start for PPO.

Why: the trained agent kept finding the "loiter safely" local optimum instead of
committing to an intercept. The heuristic in core/heuristic.py already solves the
task (~85-100% across the curriculum), so we use it as an EXPERT: roll it out,
collect (observation, action) pairs, and supervised-train a PPO policy to imitate
it. That drops the policy straight into the "pursue directly" basin. Then fine-
tune with RL:

    python pretrain_bc.py --episodes 400 --epochs 30 --target-frac 0.6
    python train_ppo.py  --continuous --resume          # continue the curriculum
    # (or)  train_ppo.py --continuous --resume --no-curriculum   # straight to full evade

The saved file is ppo_policy.zip with the SAME architecture as train_ppo, so
PPO.load() in --resume picks it up unchanged.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from stable_baselines3 import PPO

from gym_env import DroneGymEnv
from drone_core import SimulationConfig
from heuristic import expert_action
from train_ppo import HYPERPARAMS, MODEL_PATH, make_vec, evaluate


def collect(config, episodes, evasion, n_obstacles, perception, pkw, seed):
    """Roll out the expert; return (obs[N,D], act[N,2]) float32 arrays."""
    env = DroneGymEnv(evasion=evasion, closing_scale=0.10, config=config, seed=seed,
                      n_obstacles=n_obstacles, perception=perception,
                      perception_kwargs=pkw, continuous=True)
    c = env.cfg
    X, Y = [], []
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        sim = env.env.sim
        ptx, pty = sim.target.x, sim.target.y
        term = trunc = False
        info = {}
        while not (term or trunc):
            tvx, tvy = sim.target.x - ptx, sim.target.y - pty   # evader velocity
            ptx, pty = sim.target.x, sim.target.y
            th, tn = expert_action(sim.player, sim.target, tvx, tvy, sim.obstacles,
                                   c.top_speed, c.thrust_power, c.turn_speed)
            a = np.array([th, tn], dtype=np.float32)
            X.append(np.asarray(obs, dtype=np.float32)); Y.append(a)
            obs, _, term, trunc, info = env.step(a)
        wins += 1 if info.get("success") else 0
    print(f"  collected {len(X)} transitions from {episodes} expert episodes "
          f"(expert success {wins/episodes:.0%})")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def behaviour_clone(model, X, Y, epochs, batch_size, lr):
    """Supervised max-likelihood BC: maximise the policy's log-prob of expert
    actions. Robust across SB3 versions (uses policy.evaluate_actions)."""
    policy = model.policy
    policy.train()
    dev = model.device
    Xt = torch.as_tensor(X, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(Xt)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            _, log_prob, entropy = policy.evaluate_actions(Xt[idx], Yt[idx])
            loss = -log_prob.mean() - 1e-3 * entropy.mean()   # NLL + tiny entropy bonus
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  bc_loss {tot/n:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400, help="expert rollout episodes")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-frac", type=float, default=0.6,
                    help="evader speed for the BC data (mid difficulty is a good teacher)")
    ap.add_argument("--target-scale", type=float, default=1.0)
    ap.add_argument("--no-evasion", action="store_true", help="collect vs a non-fleeing target")
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--perception", choices=["none", "rays", "slots"], default="none")
    ap.add_argument("--rays", type=int, default=16)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pkw = {"n_rays": args.rays} if args.perception == "rays" else \
          {"k": args.slots} if args.perception == "slots" else {}

    config = SimulationConfig(target_speed_frac=args.target_frac * args.target_scale)
    evasion = not args.no_evasion

    print(f"Rolling out expert (target {config.target_stats['target_speed']:.1f}px/f, "
          f"evasion={evasion}, obstacles={args.obstacles}) ...")
    X, Y = collect(config, args.episodes, evasion, args.obstacles, args.perception, pkw, args.seed)

    # Build a PPO model with the SAME architecture train_ppo uses, so --resume loads it.
    venv = make_vec(evasion=evasion, closing=0.10, n_envs=1, seed=args.seed, config=config,
                    n_obstacles=args.obstacles, perception=args.perception,
                    perception_kwargs=pkw, continuous=True)
    model = PPO(env=venv, seed=args.seed, **HYPERPARAMS)

    print(f"Behaviour-cloning {len(X)} samples for {args.epochs} epochs ...")
    behaviour_clone(model, X, Y, args.epochs, args.batch_size, args.lr)

    model.save(MODEL_PATH)
    print(f"saved BC policy -> {MODEL_PATH}.zip")

    print("\n--- BC policy eval (before any RL) ---")
    evaluate(model, config=config, evasion=evasion, n_obstacles=args.obstacles,
             perception=args.perception, perception_kwargs=pkw, continuous=True)


if __name__ == "__main__":
    main()
