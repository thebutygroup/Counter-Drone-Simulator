"""
server.py
=========
WebSocket bridge between the browser viewer and the simulation.

Two modes:
  python server.py            -> human play (arrow keys, as before)
  python server.py --agent    -> a trained RL policy flies the player drone
                                 so you can WATCH it intercept in the browser.
                                 (Run qlearn.py first to produce q_policy.npz.)
"""

import asyncio
import json
import sys

import drone_core
from drone_core import DroneSimulator


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
                    data['reset'] = False
                actions = data
            except asyncio.TimeoutError:
                pass
            frame = sim.step(actions)
            await websocket.send(json.dumps(frame))
    except Exception:
        pass


def make_agent_handler():
    """Build a handler where a trained Q-policy drives the player drone."""
    from qlearn import GreedyAgent          # imported lazily so human mode needs no policy
    from drone_env import DroneInterceptEnv, ACTIONS

    agent = GreedyAgent.load("q_policy.npz")

    async def agent_handler(websocket):
        env = DroneInterceptEnv()
        obs = env.reset()
        try:
            while True:
                # Let the client send only resets in agent mode (input ignored).
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=1 / 60.0)
                    if json.loads(msg).get('reset'):
                        obs = env.reset()
                except asyncio.TimeoutError:
                    pass

                action_idx = agent.act(obs)
                obs, _reward, done, _info = env.step(action_idx)
                frame = env.render_frame()
                ctrl = ACTIONS[action_idx]
                frame["ctrl"] = {"thrust": bool(ctrl.get("thrust")),
                                 "reverse": bool(ctrl.get("reverse"))}
                await websocket.send(json.dumps(frame))
                if done:
                    await asyncio.sleep(0.4)   # brief pause so a catch is visible
                    obs = env.reset()
                await asyncio.sleep(1 / 60.0)  # pace to ~60 FPS for the viewer
        except Exception:
            pass

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
