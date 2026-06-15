"""
watch_train.py  (stage 1: 2D profile view)
==========================================
Train the interceptor while WATCHING it in the browser.

Trains headless (fast); every WATCH_EVERY episodes it plays one greedy episode
to your browser viewer with a HUD (phase / episode / rolling success / epsilon /
outcome). The on-screen SPEED SLIDER scales playback live. When training
finishes it saves q_policy.npz and loops final demos.

Run:
    python watch_train.py        # then open index.html
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
from qlearn import build_discretizer, GreedyAgent

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(HERE, "q_policy.npz")

# --- watch / training schedule ----------------------------------------------
TOTAL = 6000          # total training episodes
WATCH_EVERY = 40      # stream a live demo every N training episodes (lower = more demos)
ALPHA, GAMMA = 0.2, 0.99

# 3-phase curriculum: (label, fraction, evasion, closing_scale, eps_start)
PHASES = [
    ("SURVIVE", 0.25, False, 0.0,  1.0),
    ("CATCH",   0.35, False, 0.10, 0.5),
    ("EVADE",   0.40, True,  0.10, 0.4),
]


def train_one_episode(env, Q, disc, eps):
    obs = env.reset()
    s = disc.index(obs)
    done = False
    while not done:
        a = np.random.randint(N_ACTIONS) if np.random.rand() < eps else int(np.argmax(Q[s]))
        obs2, r, done, info = env.step(a)
        s2 = disc.index(obs2)
        Q[s, a] += ALPHA * ((r if done else r + GAMMA * np.max(Q[s2])) - Q[s, a])
        s = s2
    return info


async def stream_demo(ws, Q, disc, evasion, meta, state):
    """Play one GREEDY episode at a frame rate scaled by the live speed slider."""
    env = DroneInterceptEnv(evasion=evasion, seed=random.randint(0, 1 << 30))
    obs = env.reset()
    done = False
    info = {}
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


async def handler(ws):
    disc = build_discretizer()
    Q = np.zeros((disc.n_states, N_ACTIONS), dtype=np.float32)
    recent, ep_global = [], 0
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
    print("Browser connected -- training has begun. Watch the demos roll in.")

    try:
        for name, frac, evasion, closing, eps0 in PHASES:
            n = int(TOTAL * frac)
            drone_env.CLOSING_SCALE = closing
            env = DroneInterceptEnv(evasion=evasion, seed=hash(name) & 0xffff)

            for i in range(n):
                eps = max(0.05, eps0 * (1 - i / max(1, n)))
                info = train_one_episode(env, Q, disc, eps)
                recent.append(1.0 if info["success"] else 0.0)
                recent = recent[-200:]
                ep_global += 1

                if ep_global % WATCH_EVERY == 0:
                    meta = {"phase": name, "episode": ep_global, "total": TOTAL,
                            "success": round(float(np.mean(recent)), 3),
                            "eps": round(eps, 2)}
                    await stream_demo(ws, Q, disc, evasion, meta, state)
                elif i % 40 == 0:
                    await asyncio.sleep(0)   # yield so the speed reader stays live

        GreedyAgent(Q, disc).save(POLICY_PATH)
        print(f"Training done -> saved {POLICY_PATH}. Looping final demos.")
        while True:
            meta = {"phase": "DONE", "episode": TOTAL, "total": TOTAL,
                    "success": round(float(np.mean(recent)), 3), "eps": 0.0}
            await stream_demo(ws, Q, disc, True, meta, state)

    except websockets.exceptions.ConnectionClosed:
        print("Browser disconnected -- stopping this training run.")
    finally:
        reader_task.cancel()


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WATCH-TRAIN server on ws://localhost:8765  -- now open index.html")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
