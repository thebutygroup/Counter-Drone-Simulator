"""
server.py  (stage 1: 2D profile view)
=====================================
WebSocket bridge between the browser viewer and the simulation.

Modes:
  python server.py            -> human play (arrow keys)
  python server.py --agent    -> a trained RL policy flies the drone
                                 (run train.py or watch_train.py first)

The on-screen speed slider scales playback in --agent mode.
"""

import asyncio
import json
import os
import sys

# --- make the shared engine in ../core importable, regardless of cwd ---------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from drone_core import DroneSimulator

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(HERE, "q_policy.npz")


async def human_handler(websocket):
    """Fresh state per connection -> every reload is a true reset."""
    sim = DroneSimulator()
    actions = {"thrust": False, "reverse": False, "left": False, "right": False}
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1 / 60.0)
                data = json.loads(message)
                if data.get('reset'):
                    sim.reset()
                # Only treat a message as control input if it actually carries
                # key state -- the speed slider sends {"speed": x} and must NOT
                # blank out the controls.
                if 'thrust' in data:
                    actions = data
            except asyncio.TimeoutError:
                pass
            frame = sim.step(actions)
            await websocket.send(json.dumps(frame))
    except Exception:
        pass


def make_agent_handler():
    """Build a handler where a trained Q-policy drives the player drone."""
    from qlearn import GreedyAgent
    from drone_env import DroneInterceptEnv, ACTIONS

    agent = GreedyAgent.load(POLICY_PATH)

    async def agent_handler(websocket):
        env = DroneInterceptEnv()
        obs = env.reset()
        state = {"speed": 1.0}

        async def reader():
            try:
                async for msg in websocket:
                    d = json.loads(msg)
                    if "speed" in d:
                        state["speed"] = max(0.1, float(d["speed"]))
                    if d.get("reset"):
                        state["do_reset"] = True
            except Exception:
                pass

        reader_task = asyncio.create_task(reader())
        try:
            while True:
                if state.pop("do_reset", False):
                    obs = env.reset()
                action_idx = agent.act(obs)
                obs, _r, done, _info = env.step(action_idx)
                frame = env.render_frame()
                ctrl = ACTIONS[action_idx]
                frame["ctrl"] = {"thrust": bool(ctrl.get("thrust")),
                                 "reverse": bool(ctrl.get("reverse"))}
                await websocket.send(json.dumps(frame))
                if done:
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
    agent_mode = "--agent" in sys.argv
    handler = make_agent_handler() if agent_mode else human_handler
    label = "AGENT" if agent_mode else "HUMAN"
    async with websockets.serve(handler, "localhost", 8765):
        print(f"Simulation server started on ws://localhost:8765  [{label} mode]")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
