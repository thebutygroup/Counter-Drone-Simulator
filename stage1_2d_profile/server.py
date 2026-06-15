"""
server.py  (stage 1: 2D profile view)
=====================================
WebSocket bridge between the browser viewer and the simulation.

Modes:
  python server.py                  -> human play (arrow keys)
  python server.py --agent          -> watch the TABULAR policy (q_policy.npz)
  python server.py --agent --dqn    -> watch the NEURAL policy (dqn_policy.zip)
  python server.py --agent --batch 50   -> show "/50" as the batch denominator

In agent mode the browser HUD shows live batch stats (episode count + cumulative
success / crash rate). The Playback slider scales speed.
"""

import argparse
import asyncio
import json
import math
import os
import sys

# --- make the shared engine in ../core importable, regardless of cwd ---------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from drone_core import DroneSimulator

HERE = os.path.dirname(os.path.abspath(__file__))
Q_POLICY = os.path.join(HERE, "q_policy.npz")
DQN_POLICY = os.path.join(HERE, "dqn_policy")     # SB3 appends .zip


async def human_handler(websocket):
    sim = DroneSimulator()
    actions = {"thrust": False, "reverse": False, "left": False, "right": False}
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1 / 60.0)
                data = json.loads(message)
                if data.get('reset'):
                    sim.reset()
                if 'thrust' in data:        # ignore non-control msgs (e.g. speed)
                    actions = data
            except asyncio.TimeoutError:
                pass
            frame = sim.step(actions)
            await websocket.send(json.dumps(frame))
    except Exception:
        pass


def load_policy(use_dqn):
    """Return (act_fn, name). act_fn(raw_obs) -> action index."""
    if use_dqn:
        if not os.path.exists(DQN_POLICY + ".zip"):
            sys.exit(f"No DQN model at {DQN_POLICY}.zip -- train it with train_dqn.py first.")
        from stable_baselines3 import DQN
        model = DQN.load(DQN_POLICY)

        def act(obs):
            o = obs.astype("float32").copy()
            o[2] = (o[2] + math.pi) % (2 * math.pi) - math.pi    # match training obs
            a, _ = model.predict(o, deterministic=True)
            return int(a)
        return act, "DQN"
    else:
        if not os.path.exists(Q_POLICY):
            sys.exit(f"No tabular model at {Q_POLICY} -- train it with train.py first.")
        from qlearn import GreedyAgent
        agent = GreedyAgent.load(Q_POLICY)
        return agent.act, "TABULAR"


def make_agent_handler(use_dqn, batch):
    from drone_env import DroneInterceptEnv, ACTIONS
    act, name = load_policy(use_dqn)
    total = batch if batch > 0 else "∞"

    async def agent_handler(websocket):
        env = DroneInterceptEnv()
        obs = env.reset()
        state = {"speed": 1.0}
        st = {"ep": 0, "win": 0, "crash": 0, "timeout": 0}

        async def reader():
            try:
                async for msg in websocket:
                    d = json.loads(msg)
                    if "speed" in d:
                        state["speed"] = max(0.1, float(d["speed"]))
            except Exception:
                pass
        reader_task = asyncio.create_task(reader())

        def hud(result=None):
            rate = (st["win"] / st["ep"]) if st["ep"] else 0.0
            m = {"phase": name, "episode": st["ep"], "total": total,
                 "success": round(rate, 3)}            # eps omitted -> HUD shows '-'
            if result:
                m["result"] = result
            return m

        try:
            while True:
                a = act(obs)
                obs, _r, done, info = env.step(a)
                frame = env.render_frame()
                ctrl = ACTIONS[a]
                frame["ctrl"] = {"thrust": bool(ctrl.get("thrust")),
                                 "reverse": bool(ctrl.get("reverse"))}
                frame["train"] = hud()
                await websocket.send(json.dumps(frame))

                if done:
                    st["ep"] += 1
                    if info["success"]:
                        st["win"] += 1; res = "INTERCEPT"
                    elif info["crashed"]:
                        st["crash"] += 1; res = "CRASH"
                    else:
                        st["timeout"] += 1; res = "TIMEOUT"
                    frame["train"] = hud(res)
                    await websocket.send(json.dumps(frame))

                    if batch > 0 and st["ep"] % batch == 0:
                        print(f"[{name}] batch of {batch}: "
                              f"success={st['win']/st['ep']:.1%} "
                              f"crash={st['crash']/st['ep']:.1%} (n={st['ep']})")

                    await asyncio.sleep(0.4 / state["speed"])
                    obs = env.reset()

                await asyncio.sleep((1 / 60.0) / state["speed"])
        except Exception:
            pass
        finally:
            reader_task.cancel()

    return agent_handler


async def main():
    import websockets
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--dqn", action="store_true", help="use dqn_policy.zip (implies --agent)")
    ap.add_argument("--batch", type=int, default=0, help="batch size shown as the denominator")
    args = ap.parse_args()

    agent_mode = args.agent or args.dqn
    if agent_mode:
        handler = make_agent_handler(args.dqn, args.batch)
        label = "AGENT/DQN" if args.dqn else "AGENT/TABULAR"
    else:
        handler = human_handler
        label = "HUMAN"

    async with websockets.serve(handler, "localhost", 8765):
        print(f"Simulation server started on ws://localhost:8765  [{label} mode]")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
