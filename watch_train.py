"""
watch_train.py
==============
Train the interceptor while WATCHING it in the browser.

It trains headless (fast), and every WATCH_EVERY episodes it plays one greedy
episode to your browser viewer at real-time speed, with a HUD showing the phase,
episode count, rolling success rate, and the outcome of each demo. So you see
the agent visibly get better (and watch the difficulty spike when evasion turns
on). When training finishes it saves q_policy.npz and loops final demos forever.

Run:
    python watch_train.py        # then open index.html in your browser
"""

import asyncio
import json
import random

import numpy as np
import websockets

import drone_env
from drone_env import DroneInterceptEnv, N_ACTIONS, ACTIONS
from qlearn import build_discretizer, GreedyAgent

# --- watch / training schedule ----------------------------------------------
TOTAL = 6000          # total training episodes
WATCH_EVERY = 120     # stream one live demo episode every N training episodes
ALPHA, GAMMA = 0.2, 0.99

# 3-phase curriculum: (label, fraction, evasion, closing_scale, eps_start)
PHASES = [
    ("SURVIVE", 0.25, False, 0.0,  1.0),
    ("CATCH",   0.35, False, 0.10, 0.5),
    ("EVADE",   0.40, True,  0.10, 0.4),
]


def train_one_episode(env, Q, disc, eps):
    """One headless Q-learning episode. Returns the terminal info dict."""
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


async def stream_demo(ws, Q, disc, evasion, meta):
    """Play one GREEDY episode to the browser at ~60fps."""
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
        await asyncio.sleep(1 / 60)

    # Flash the outcome on the last frame, then pause so it's readable.
    result = "INTERCEPT" if info.get("success") else (
        "CRASH" if info.get("crashed") else "TIMEOUT")
    frame["train"] = {**meta, "result": result}
    await ws.send(json.dumps(frame))
    await asyncio.sleep(0.7)


async def handler(ws):
    disc = build_discretizer()
    Q = np.zeros((disc.n_states, N_ACTIONS), dtype=np.float32)
    recent = []
    ep_global = 0
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
                    await stream_demo(ws, Q, disc, evasion, meta)
                elif i % 40 == 0:
                    await asyncio.sleep(0)   # yield so the socket stays alive

        GreedyAgent(Q, disc).save("q_policy.npz")
        print("Training done -> saved q_policy.npz. Looping final demos.")
        while True:
            meta = {"phase": "DONE", "episode": TOTAL, "total": TOTAL,
                    "success": round(float(np.mean(recent)), 3), "eps": 0.0}
            await stream_demo(ws, Q, disc, True, meta)

    except websockets.exceptions.ConnectionClosed:
        print("Browser disconnected -- stopping this training run.")


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WATCH-TRAIN server on ws://localhost:8765  -- now open index.html")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
