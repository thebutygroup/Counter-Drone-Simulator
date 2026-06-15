"""
benchmark.py  (stage 1: 2D profile view)
========================================
Apples-to-apples comparison of the TABULAR Q agent vs the NEURAL (DQN) agent.

Both agents are evaluated on the *same seeded episodes* (identical target spawns
and physics), so any difference is the policy alone. For each agent it records:

  ACCURACY    success rate, crash rate, timeout rate, avg catch time, avg impact
  SIMPLICITY  parameter count, on-disk size, inference latency, decisions/sec
  SCALE       how much training produced it (episodes / timesteps)
  EFFICIENCY  success per 100k params (accuracy-per-complexity)

Results are appended to benchmark_results.csv (so you can track many model
variants over time) and printed as a table + raw CSV.

Run (whichever models are present are benchmarked):
    python benchmark.py                       # 500 eval episodes per regime
    python benchmark.py --episodes 1000 --label dqn-256x256-1.5M
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from drone_env import DroneInterceptEnv

HERE = os.path.dirname(os.path.abspath(__file__))
Q_POLICY = os.path.join(HERE, "q_policy.npz")
Q_CKPT = os.path.join(HERE, "checkpoint.npz")
DQN_POLICY = os.path.join(HERE, "dqn_policy.zip")
RESULTS_CSV = os.path.join(HERE, "benchmark_results.csv")

FIELDS = ["timestamp", "agent", "label", "regime", "episodes",
          "success_rate", "crash_rate", "timeout_rate",
          "avg_catch_frames", "avg_impact_speed",
          "params", "disk_kb", "infer_us", "decisions_per_sec",
          "train_size", "train_unit", "success_per_100k_params"]


def _wrap_obs(obs):
    """Match the DQN's training-time observation (heading wrapped to [-pi, pi])."""
    obs = np.asarray(obs, dtype=np.float32).copy()
    obs[2] = (obs[2] + math.pi) % (2 * math.pi) - math.pi
    return obs


# ---------------------------------------------------------------------------
# Agent loaders -> each returns (policy_fn, sizeinfo) or None if unavailable.
# policy_fn(raw_obs) -> action index
# ---------------------------------------------------------------------------
def load_tabular():
    if not os.path.exists(Q_POLICY):
        return None
    from qlearn import GreedyAgent
    agent = GreedyAgent.load(Q_POLICY)
    Q = agent.Q
    train_size, train_unit = "n/a", "episodes"
    if os.path.exists(Q_CKPT):
        try:
            d = np.load(Q_CKPT, allow_pickle=True)
            train_size = int(d["episodes_done"]) if "episodes_done" in d else "n/a"
        except Exception:
            pass
    size = {
        "params": int(np.count_nonzero(Q)),          # nonzero cells = effective size
        "params_total": int(Q.size),
        "disk_kb": round(os.path.getsize(Q_POLICY) / 1024, 1),
        "train_size": train_size, "train_unit": train_unit,
    }
    return agent.act, size


def load_dqn():
    if not os.path.exists(DQN_POLICY):
        return None
    from stable_baselines3 import DQN          # lazy: only needed if model present
    model = DQN.load(DQN_POLICY)
    n_params = int(sum(p.numel() for p in model.policy.parameters()))
    size = {
        "params": n_params, "params_total": n_params,
        "disk_kb": round(os.path.getsize(DQN_POLICY) / 1024, 1),
        "train_size": int(getattr(model, "num_timesteps", 0)), "train_unit": "timesteps",
    }

    def policy_fn(obs):
        action, _ = model.predict(_wrap_obs(obs), deterministic=True)
        return int(action)

    return policy_fn, size


# ---------------------------------------------------------------------------
# Evaluation on identical seeded episodes
# ---------------------------------------------------------------------------
def run_eval(policy_fn, evasion, episodes):
    wins = crashes = timeouts = 0
    catch_frames, impacts = [], []
    for s in range(episodes):
        env = DroneInterceptEnv(evasion=evasion, seed=s)   # same seed -> same scenario
        obs = env.reset()
        done = False
        info = {}
        while not done:
            obs, _, done, info = env.step(policy_fn(obs))
        if info["success"]:
            wins += 1; catch_frames.append(info["steps"]); impacts.append(info["impact_speed"])
        elif info["crashed"]:
            crashes += 1
        else:
            timeouts += 1
    n = float(episodes)
    return {
        "success_rate": wins / n,
        "crash_rate": crashes / n,
        "timeout_rate": timeouts / n,
        "avg_catch_frames": round(float(np.mean(catch_frames)), 1) if catch_frames else "",
        "avg_impact_speed": round(float(np.mean(impacts)), 2) if impacts else "",
    }


def measure_latency(policy_fn, samples=2000):
    """Time per-decision inference on representative observations."""
    obs_pool = []
    env = DroneInterceptEnv(evasion=True, seed=12345)
    obs = env.reset(); done = False
    while len(obs_pool) < samples:
        obs_pool.append(obs)
        if done:
            obs = env.reset(); done = False
        else:
            obs, _, done, _ = env.step(policy_fn(obs))
    t0 = time.perf_counter()
    for o in obs_pool:
        policy_fn(o)
    dt = time.perf_counter() - t0
    return round(dt / samples * 1e6, 2), int(samples / dt)   # us/decision, decisions/sec


# ---------------------------------------------------------------------------
def persist(rows):
    is_new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def print_table(rows):
    cols = [("agent", 9), ("regime", 11), ("success_rate", 8), ("crash_rate", 7),
            ("avg_catch_frames", 9), ("params", 10), ("disk_kb", 8),
            ("infer_us", 9), ("decisions_per_sec", 11), ("success_per_100k_params", 10)]
    header = "  ".join(name.replace("_", " ")[:w].ljust(w) for name, w in cols)
    print("\n" + header); print("-" * len(header))
    for r in rows:
        line = []
        for name, w in cols:
            v = r.get(name, "")
            if name.endswith("_rate"):
                v = f"{v:.1%}"
            line.append(str(v)[:w].ljust(w))
        print("  ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--label", default="", help="tag this run (e.g. dqn-64x64-600k)")
    args = ap.parse_args()

    agents = []
    tab = load_tabular()
    if tab:
        agents.append(("tabular", *tab))
    else:
        print("(no q_policy.npz -> skipping tabular)")
    dqn = load_dqn()
    if dqn:
        agents.append(("dqn", *dqn))
    else:
        print("(no dqn_policy.zip -> skipping DQN)")

    if not agents:
        sys.exit("No models found to benchmark.")

    ts = datetime.now().isoformat(timespec="seconds")
    rows = []
    for name, policy_fn, size in agents:
        print(f"\nBenchmarking '{name}' over {args.episodes} episodes/regime ...")
        us, dps = measure_latency(policy_fn)
        params = size["params"]
        for regime, evasion in [("evade", True), ("non-evade", False)]:
            m = run_eval(policy_fn, evasion, args.episodes)
            spp = round(m["success_rate"] / (params / 100_000), 3) if params else ""
            rows.append({
                "timestamp": ts, "agent": name, "label": args.label, "regime": regime,
                "episodes": args.episodes, **m,
                "params": params, "disk_kb": size["disk_kb"],
                "infer_us": us, "decisions_per_sec": dps,
                "train_size": size["train_size"], "train_unit": size["train_unit"],
                "success_per_100k_params": spp,
            })

    persist(rows)
    print_table(rows)
    print(f"\nAppended {len(rows)} rows to {RESULTS_CSV}\n")
    print("=== benchmark_results.csv (full) ===")
    with open(RESULTS_CSV) as f:
        print(f.read().strip())


if __name__ == "__main__":
    main()
