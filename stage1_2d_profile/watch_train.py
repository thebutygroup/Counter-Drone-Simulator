"""
watch_train.py  (stage 1: 2D profile view)
==========================================
Train the interceptor while WATCHING it in the browser -- now with the same
checkpoint / resume / metrics system as train.py.

Behaviour:
  * If checkpoint.npz exists, it RESUMES from it (continues the most up-to-date
    policy in the hard EVADE regime). Use --fresh to start a new curriculum.
  * Writes checkpoint.npz + metrics.csv + q_policy.npz periodically, so a watched
    run is crash-safe and resumable (by this script OR train.py).
  * Streams a live greedy demo to the browser every WATCH_EVERY episodes, with a
    HUD and an outcome flash. The on-screen Playback slider scales demo speed.

Run:
    python watch_train.py          # resume if a checkpoint exists, else fresh
    python watch_train.py --fresh  # ignore any checkpoint, start a new curriculum
Then open index.html.
"""

import asyncio
import json
import os
import random
import sys

import numpy as np

# --- make the shared engine in ../core importable, regardless of cwd ---------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

import websockets
import drone_env
from drone_env import DroneInterceptEnv, N_ACTIONS, ACTIONS
from qlearn import (build_discretizer, load_checkpoint, save_checkpoint,
                    _log_metrics_row, GreedyAgent)

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(HERE, "q_policy.npz")
CKPT = os.path.join(HERE, "checkpoint.npz")
METRICS = os.path.join(HERE, "metrics.csv")

# --- schedule ---------------------------------------------------------------
TOTAL = 6000             # episodes to run this invocation
WATCH_EVERY = 40         # stream a live demo every N episodes
CHECKPOINT_EVERY = 250   # write checkpoint + metrics + policy every N episodes
ALPHA, GAMMA = 0.2, 0.99

# fresh 3-phase curriculum: (label, fraction, evasion, closing_scale, eps_start)
FRESH_PHASES = [
    ("SURVIVE", 0.25, False, 0.0,  1.0),
    ("CATCH",   0.35, False, 0.10, 0.5),
    ("EVADE",   0.40, True,  0.10, 0.4),
]


def train_one_episode(env, Q, disc, eps):
    obs = env.reset()
    s = disc.index(obs)
    done, ep_r = False, 0.0
    while not done:
        a = np.random.randint(N_ACTIONS) if np.random.rand() < eps else int(np.argmax(Q[s]))
        obs2, r, done, info = env.step(a)
        s2 = disc.index(obs2)
        Q[s, a] += ALPHA * ((r if done else r + GAMMA * np.max(Q[s2])) - Q[s, a])
        s, ep_r = s2, ep_r + r
    return info, ep_r


async def stream_demo(ws, Q, disc, evasion, meta, state):
    env = DroneInterceptEnv(evasion=evasion, seed=random.randint(0, 1 << 30))
    obs = env.reset()
    done, info = False, {}
    while not done:
        a = int(np.argmax(Q[disc.index(obs)]))
        obs, _, done, info = env.step(a)
        frame = env.render_frame()
        ctrl = ACTIONS[a]
        frame["ctrl"] = {"thrust": bool(ctrl.get("thrust")),
                         "reverse": bool(ctrl.get("reverse"))}
        frame["train"] = meta
        await ws.send(json.dumps(frame))
        await asyncio.sleep((1 / 60) / state["speed"])
    result = "INTERCEPT" if info.get("success") else (
        "CRASH" if info.get("crashed") else "TIMEOUT")
    frame["train"] = {**meta, "result": result}
    await ws.send(json.dumps(frame))
    await asyncio.sleep(0.7 / state["speed"])


def _plan():
    """Decide fresh-vs-resume. Returns (phases, Q, disc, history_lists, offset)."""
    if os.path.exists(CKPT) and "--fresh" not in sys.argv:
        Q, disc, hist, meta = load_checkpoint(CKPT)
        offset = meta["episodes_done"]
        print(f"Resuming from checkpoint at episode {offset} "
              f"({len(hist.get('success', []))} eps of history). "
              f"Use --fresh to start over.")
        succ = list(hist.get("success", []))
        crash = list(hist.get("crash", []))
        rew = list(hist.get("reward", []))
        phases = [("EVADE+", TOTAL, True, 0.10, 0.3)]
        return phases, Q, disc, (succ, crash, rew), offset
    else:
        disc = build_discretizer()
        Q = np.zeros((disc.n_states, N_ACTIONS), dtype=np.float32)
        phases = [(name, int(TOTAL * f), ev, cs, e0)
                  for name, f, ev, cs, e0 in FRESH_PHASES]
        print("Fresh curriculum (no checkpoint found or --fresh given).")
        return phases, Q, disc, ([], [], []), 0


def _save_all(Q, disc, ep_global, succ, crash, rew, phase, eps):
    history = {"success": succ, "crash": crash, "reward": rew}
    save_checkpoint(CKPT, Q, disc, ep_global, history, phase=phase)
    GreedyAgent(Q, disc).save(POLICY)
    w = CHECKPOINT_EVERY
    _log_metrics_row(METRICS, ep_global, phase,
                     {"success": succ[-w:], "crash": crash[-w:],
                      "reward": rew[-w:], "eps": eps})


async def handler(ws):
    phases, Q, disc, (succ, crash, rew), offset = _plan()
    state = {"speed": 1.0}

    async def reader():
        try:
            async for msg in ws:
                d = json.loads(msg)
                if "speed" in d:
                    state["speed"] = max(0.1, float(d["speed"]))
        except Exception:
            pass

    reader_task = asyncio.create_task(reader())
    ep_global = offset
    last_phase = phases[-1][0]
    last_evasion = phases[-1][2]

    try:
        for name, n, evasion, closing, eps0 in phases:
            drone_env.CLOSING_SCALE = closing
            env = DroneInterceptEnv(evasion=evasion, seed=random.randint(0, 1 << 30))
            for i in range(n):
                eps = max(0.05, eps0 * (1 - i / max(1, n)))
                info, ep_r = train_one_episode(env, Q, disc, eps)
                succ.append(1.0 if info["success"] else 0.0)
                crash.append(1.0 if info["crashed"] else 0.0)
                rew.append(ep_r)
                ep_global += 1

                streamed = False
                if ep_global % WATCH_EVERY == 0:
                    meta = {"phase": name, "episode": ep_global, "total": offset + sum(p[1] for p in phases),
                            "success": round(float(np.mean(succ[-200:])), 3),
                            "eps": round(eps, 2)}
                    await stream_demo(ws, Q, disc, evasion, meta, state)
                    streamed = True

                if ep_global % CHECKPOINT_EVERY == 0:
                    _save_all(Q, disc, ep_global, succ, crash, rew, name, eps)

                if not streamed and i % 40 == 0:
                    await asyncio.sleep(0)

        if ep_global % CHECKPOINT_EVERY != 0:
            _save_all(Q, disc, ep_global, succ, crash, rew, last_phase, 0.05)
        print(f"Training done -> saved checkpoint/policy/metrics. Looping demos.")
        while True:
            meta = {"phase": "DONE", "episode": ep_global, "total": ep_global,
                    "success": round(float(np.mean(succ[-200:])), 3), "eps": 0.0}
            await stream_demo(ws, Q, disc, last_evasion, meta, state)

    except websockets.exceptions.ConnectionClosed:
        print("Browser disconnected -- saving checkpoint and stopping.")
        _save_all(Q, disc, ep_global, succ, crash, rew, "interrupted", 0.05)
    finally:
        reader_task.cancel()


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WATCH-TRAIN server on ws://localhost:8765  -- now open index.html")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())