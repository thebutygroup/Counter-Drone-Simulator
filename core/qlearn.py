"""
qlearn.py
=========
Tabular Q-learning interceptor.

Why tabular (not DQN)? This box has numpy but no torch/gymnasium, and for a
low-dimensional pursuit task a discretised Q-table is a strong, fast,
fully-interpretable baseline. The environment (drone_env.py) already speaks the
Gym API, so swapping in a neural agent later is a drop-in replacement.

Run:
    python qlearn.py                  # curriculum train, save q_policy.npz + curve PNG
    python qlearn.py --episodes 4000  # override length
Then watch it:
    python server.py --agent          # the policy flies the drone in the browser
"""

import argparse
import csv
import json
import math
import os
import tempfile

import numpy as np

from drone_env import DroneInterceptEnv, N_ACTIONS
import drone_core


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
# Two distinct things get stored:
#   * q_policy.npz   -> the deployable POLICY (Q-table + config). server.py loads this.
#   * checkpoint.npz -> full TRAINING STATE (policy + episodes_done + history),
#                       written periodically so a crash/Ctrl-C is recoverable and
#                       you can RESUME instead of starting over.
#   * metrics.csv    -> append-only progress log (success/crash/reward over time),
#                       so you keep a durable record across runs (open it in Excel).
# Saves are ATOMIC (write temp -> rename) so an interrupt can't corrupt a file.

def _atomic_savez(path, **arrays):
    folder = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".npz")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)          # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_checkpoint(path, Q, disc, episodes_done, history, phase=""):
    """Full training state -> resumable."""
    _atomic_savez(
        path,
        Q=Q,
        config=json.dumps(disc.config()),
        episodes_done=np.int64(episodes_done),
        phase=str(phase),
        success=np.asarray(history.get("success", []), dtype=np.float32),
        crash=np.asarray(history.get("crash", []), dtype=np.float32),
        reward=np.asarray(history.get("reward", []), dtype=np.float32),
    )


def load_checkpoint(path):
    """Returns (Q, disc, history, meta) for resuming."""
    d = np.load(path, allow_pickle=True)
    disc = Discretizer.from_config(json.loads(str(d["config"])))
    history = {k: list(map(float, d[k])) for k in ("success", "crash", "reward") if k in d}
    meta = {"episodes_done": int(d["episodes_done"]) if "episodes_done" in d else 0,
            "phase": str(d["phase"]) if "phase" in d else ""}
    return d["Q"].copy(), disc, history, meta


def _log_metrics_row(path, episode, phase, window):
    """Append one rolling-summary row to a CSV (header written on first use)."""
    is_new = not os.path.exists(path)
    s = window["success"]; c = window["crash"]; r = window["reward"]
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["episode", "phase", "success_rate", "crash_rate",
                        "mean_reward", "epsilon"])
        w.writerow([episode, phase,
                    round(float(np.mean(s)), 4) if s else "",
                    round(float(np.mean(c)), 4) if c else "",
                    round(float(np.mean(r)), 3) if r else "",
                    round(window["eps"], 3)])


# ---------------------------------------------------------------------------
# Discretiser: map the continuous observation -> one integer state id.
# Each spec is (observation_index, bin_edges). Tune the edges to trade off
# table size against resolution.
# ---------------------------------------------------------------------------
def _uniform_edges(nbins, lo=-math.pi, hi=math.pi):
    """Interior edges for `nbins` uniform buckets over [lo, hi]."""
    return np.linspace(lo, hi, nbins + 1)[1:-1]


def build_discretizer():
    # Fine bins for position NEAR the walls (where crashing happens) and coarse
    # in the open middle. angle/vx trimmed to keep the table ~400k states.
    wall_edges = np.array([0.10, 0.22, 0.78, 0.90])   # -> 5 buckets, dense at edges
    specs = [
        (1, _uniform_edges(16)),                              # rel_angle  -> 16
        (0, np.array([0.06, 0.10, 0.16, 0.25, 0.40])),        # dist_norm  -> 6
        (2, _uniform_edges(8)),                               # heading    -> 8
        (4, np.array([-4.0, -1.5, -0.3, 0.3, 1.5, 4.0])),     # vy         -> 7
        (3, np.array([-1.0, 1.0])),                           # vx         -> 3
        (7, wall_edges),                                      # px         -> 5
        (8, wall_edges),                                      # py         -> 5
    ]
    return Discretizer(specs)


class Discretizer:
    def __init__(self, specs):
        self.obs_idx = [oi for oi, _ in specs]
        self.edges = [np.asarray(e, dtype=np.float64) for _, e in specs]
        self.sizes = [len(e) + 1 for e in self.edges]
        self.n_states = int(np.prod(self.sizes))

    def index(self, obs):
        idx = 0
        for oi, edges, size in zip(self.obs_idx, self.edges, self.sizes):
            b = int(np.digitize(obs[oi], edges))   # 0 .. size-1
            idx = idx * size + b
        return idx

    def config(self):
        return {"obs_idx": self.obs_idx, "edges": [e.tolist() for e in self.edges]}

    @classmethod
    def from_config(cls, cfg):
        specs = list(zip(cfg["obs_idx"], cfg["edges"]))
        return cls(specs)


# ---------------------------------------------------------------------------
# Greedy agent = trained Q-table + its discretiser. Used at inference time.
# ---------------------------------------------------------------------------
class GreedyAgent:
    def __init__(self, Q, disc):
        self.Q = Q
        self.disc = disc

    def act(self, obs):
        return int(np.argmax(self.Q[self.disc.index(obs)]))

    def save(self, path):
        _atomic_savez(path, Q=self.Q, config=json.dumps(self.disc.config()))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        disc = Discretizer.from_config(json.loads(str(d["config"])))
        return cls(d["Q"], disc)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(episodes, evasion, target_speed=drone_core.TARGET_SPEED,
          alpha=0.2, gamma=0.99, eps_start=1.0, eps_end=0.05,
          Q=None, disc=None, seed=0, log_every=500, label="",
          ckpt_path=None, ckpt_every=0, metrics_csv=None,
          prior_history=None, episode_offset=0):
    """Train `episodes` episodes.

    Resumable-friendly extras:
      ckpt_path/ckpt_every : write a full checkpoint every N episodes (atomic).
      metrics_csv          : append a rolling-summary row at each checkpoint.
      prior_history        : history to PREPEND (so checkpoints stay cumulative
                             across curriculum phases / resumes).
      episode_offset       : global episode number of the first episode here.
    """
    disc = disc or build_discretizer()
    if Q is None:
        Q = np.zeros((disc.n_states, N_ACTIONS), dtype=np.float32)
    env = DroneInterceptEnv(evasion=evasion, target_speed=target_speed, seed=seed)

    ph = prior_history or {}
    successes = list(ph.get("success", []))
    crashes = list(ph.get("crash", []))
    rewards = list(ph.get("reward", []))

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * (1 - ep / max(1, episodes))
        obs = env.reset()
        s = disc.index(obs)
        done, ep_r = False, 0.0
        while not done:
            if np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[s]))
            obs2, r, done, info = env.step(a)
            s2 = disc.index(obs2)
            td_target = r if done else r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (td_target - Q[s, a])
            s, ep_r = s2, ep_r + r
        successes.append(1.0 if info["success"] else 0.0)
        crashes.append(1.0 if info["crashed"] else 0.0)
        rewards.append(ep_r)

        global_ep = episode_offset + ep + 1
        history = {"success": successes, "reward": rewards, "crash": crashes}

        if log_every and (ep + 1) % log_every == 0:
            w = log_every
            print(f"  [{label}] ep {global_ep:6d}  eps={eps:.2f}  "
                  f"success={np.mean(successes[-w:]):5.1%}  "
                  f"crash={np.mean(crashes[-w:]):5.1%}  "
                  f"meanR={np.mean(rewards[-w:]):7.2f}")

        if ckpt_every and global_ep % ckpt_every == 0:
            if ckpt_path:
                save_checkpoint(ckpt_path, Q, disc, global_ep, history, phase=label)
            if metrics_csv:
                w = ckpt_every
                _log_metrics_row(metrics_csv, global_ep, label,
                                 {"success": successes[-w:], "crash": crashes[-w:],
                                  "reward": rewards[-w:], "eps": eps})

    return Q, disc, {"success": successes, "reward": rewards, "crash": crashes}


def run_curriculum(total_episodes, seed=0, ckpt_path=None, ckpt_every=500,
                   metrics_csv=None):
    """Three stages, each building on the last:
       Phase 0 SURVIVE - no target pressure (closing reward off); just learn to
                         fly and stop crashing into walls.
       Phase 1 CATCH   - closing reward on; intercept a non-evading target.
       Phase 2 EVADE   - harden against the evading target.
    Checkpoints + metrics are written continuously if paths are given."""
    import drone_env
    disc = build_discretizer()

    n0 = int(total_episodes * 0.25)
    n1 = int(total_episodes * 0.35)
    n2 = total_episodes - n0 - n1

    common = dict(ckpt_path=ckpt_path, ckpt_every=ckpt_every, metrics_csv=metrics_csv)

    print(f"Phase 0: {n0} episodes (SURVIVE - learn to fly)")
    drone_env.CLOSING_SCALE = 0.0
    Q, disc, hist = train(n0, evasion=False, disc=disc, seed=seed, label="survive",
                          episode_offset=0, **common)

    print(f"Phase 1: {n1} episodes (CATCH - non-evading target)")
    drone_env.CLOSING_SCALE = 0.10
    Q, disc, hist = train(n1, evasion=False, Q=Q, disc=disc, seed=seed + 1,
                          eps_start=0.5, label="catch",
                          prior_history=hist, episode_offset=n0, **common)

    print(f"Phase 2: {n2} episodes (EVADE - evasion ON)")
    Q, disc, hist = train(n2, evasion=True, Q=Q, disc=disc, seed=seed + 2,
                          eps_start=0.4, label="evade",
                          prior_history=hist, episode_offset=n0 + n1, **common)

    return Q, disc, hist, n0 + n1   # split marker = evasion start


def continue_training(ckpt_path, episodes, seed=0, ckpt_every=500, metrics_csv=None):
    """Resume from a checkpoint and train MORE episodes in the hard (EVADE)
       regime. Returns (Q, disc, history, split)."""
    import drone_env
    Q, disc, hist, meta = load_checkpoint(ckpt_path)
    offset = meta["episodes_done"]
    print(f"Resuming from {ckpt_path} at episode {offset} "
          f"({len(hist.get('success', []))} episodes of history)")

    drone_env.CLOSING_SCALE = 0.10
    Q, disc, hist = train(episodes, evasion=True, Q=Q, disc=disc, seed=seed,
                          eps_start=0.3, label="evade+",
                          prior_history=hist, episode_offset=offset,
                          ckpt_path=ckpt_path, ckpt_every=ckpt_every,
                          metrics_csv=metrics_csv)
    # split marker: where evasion-era data begins is unknown post-resume, so use 0
    return Q, disc, hist, 0


def plot_history(history, split, path="training_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(x, w=100):
        x = np.asarray(x, dtype=float)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(smooth(history["success"]))
    ax[0].axvline(split, color="r", ls="--", lw=1, label="evasion on")
    ax[0].set_title("Intercept success rate (100-ep moving avg)")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("success"); ax[0].legend()
    ax[1].plot(smooth(history["reward"]))
    ax[1].axvline(split, color="r", ls="--", lw=1)
    ax[1].set_title("Episode reward (100-ep moving avg)")
    ax[1].set_xlabel("episode"); ax[1].set_ylabel("reward")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Saved {path}")


def evaluate(agent, episodes=300, evasion=True, seed=999):
    env = DroneInterceptEnv(evasion=evasion, seed=seed)
    wins, crashes, steps, impacts = 0, 0, [], []
    for _ in range(episodes):
        obs = env.reset(); done = False
        while not done:
            obs, _, done, info = env.step(agent.act(obs))
        if info["success"]:
            wins += 1; steps.append(info["steps"]); impacts.append(info["impact_speed"])
        elif info["crashed"]:
            crashes += 1
    sr = wins / episodes
    cr = crashes / episodes
    avg = np.mean(steps) if steps else float("nan")
    spd = np.mean(impacts) if impacts else float("nan")
    print(f"Eval: success={sr:.1%}  crash={cr:.1%}  over {episodes} eps  "
          f"(avg catch in {avg:.0f} frames, avg impact speed {spd:.1f} px/frame)")
    return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="q_policy.npz")
    args = ap.parse_args()

    np.random.seed(args.seed)
    Q, disc, history, split = run_curriculum(args.episodes, seed=args.seed)
    agent = GreedyAgent(Q, disc)
    agent.save(args.out)
    print(f"Saved policy -> {args.out}")
    plot_history(history, split)
    evaluate(agent, evasion=True)


if __name__ == "__main__":
    main()
