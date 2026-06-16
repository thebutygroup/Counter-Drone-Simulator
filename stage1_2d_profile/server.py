"""
server.py  (stage 1: 2D profile view)
=====================================
WebSocket bridge between the browser viewer and the simulation.

Modes:
  python server.py                  -> human play (arrow keys)
  python server.py --agent          -> watch the TABULAR policy (q_policy.npz)
  python server.py --agent --dqn    -> watch the DQN policy   (dqn_policy.zip)
  python server.py --agent --ppo    -> watch the PPO policy   (ppo_policy.zip)
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

import numpy as np

# --- make the shared engine in ../core importable, regardless of cwd ---------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from drone_core import DroneSimulator

HERE = os.path.dirname(os.path.abspath(__file__))
Q_POLICY = os.path.join(HERE, "q_policy.npz")
DQN_POLICY = os.path.join(HERE, "dqn_policy")     # SB3 appends .zip
PPO_POLICY = os.path.join(HERE, "ppo_policy")     # SB3 appends .zip


def make_human_handler(n_obstacles=0):
    async def human_handler(websocket):
        sim = DroneSimulator(n_obstacles=n_obstacles)
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
    return human_handler


def _sb3_act(model):
    """Wrap an SB3 model (DQN or PPO) into act(raw_obs) -> action.
    Applies the same heading wrap the policy saw at training time, and selects
    deterministically (greedy) for a clean watch/eval. Auto-detects a Box action
    space (continuous PPO) and returns the float [thrust, turn] vector in that
    case; otherwise returns a discrete action index."""
    from gymnasium.spaces import Box
    continuous = isinstance(model.action_space, Box)

    def act(obs):
        o = obs.astype("float32").copy()
        o[2] = (o[2] + math.pi) % (2 * math.pi) - math.pi    # match training obs
        a, _ = model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32) if continuous else int(a)
    return act, continuous


def load_policy(kind):
    """kind in {'tabular','dqn','ppo'}. Return (act_fn, name, continuous)."""
    if kind == "dqn":
        if not os.path.exists(DQN_POLICY + ".zip"):
            sys.exit(f"No DQN model at {DQN_POLICY}.zip -- train it with train_dqn.py first.")
        from stable_baselines3 import DQN
        act, cont = _sb3_act(DQN.load(DQN_POLICY))
        return act, "DQN", cont
    if kind == "ppo":
        if not os.path.exists(PPO_POLICY + ".zip"):
            sys.exit(f"No PPO model at {PPO_POLICY}.zip -- train it with train_ppo.py first.")
        from stable_baselines3 import PPO
        act, cont = _sb3_act(PPO.load(PPO_POLICY))
        return act, "PPO", cont
    # tabular
    if not os.path.exists(Q_POLICY):
        sys.exit(f"No tabular model at {Q_POLICY} -- train it with train.py first.")
    from qlearn import GreedyAgent
    agent = GreedyAgent.load(Q_POLICY)
    return agent.act, "TABULAR", False


def make_agent_handler(kind, batch):
    from drone_env import DroneInterceptEnv, ACTIONS
    act, name, continuous = load_policy(kind)
    total = batch if batch > 0 else "∞"

    async def agent_handler(websocket):
        env = DroneInterceptEnv(continuous=continuous)
        obs = env.reset()
        state = {"speed": 1.0}
        st = {"ep": 0, "win": 0, "crash": 0, "escape": 0, "timeout": 0}

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
                # Thruster-flame HUD: discrete -> look up the input dict; continuous
                # -> derive fwd/reverse from the sign of the thrust channel a[0].
                if continuous:
                    th = float(np.asarray(a).reshape(-1)[0])
                    frame["ctrl"] = {"thrust": th > 0.05, "reverse": th < -0.05}
                else:
                    ctrl = ACTIONS[a]
                    frame["ctrl"] = {"thrust": bool(ctrl.get("thrust")),
                                     "reverse": bool(ctrl.get("reverse"))}
                frame["train"] = hud()
                await websocket.send(json.dumps(frame))

                if done:
                    st["ep"] += 1
                    if info["success"]:
                        st["win"] += 1; res = "INTERCEPT"
                    elif info.get("escaped"):
                        st["escape"] += 1; res = "ESCAPE"
                    elif info["crashed"]:
                        st["crash"] += 1; res = "CRASH"
                    else:
                        st["timeout"] += 1; res = "TIMEOUT"
                    frame["train"] = hud(res)
                    await websocket.send(json.dumps(frame))

                    if batch > 0 and st["ep"] % batch == 0:
                        print(f"[{name}] batch of {batch}: "
                              f"success={st['win']/st['ep']:.1%} "
                              f"crash={st['crash']/st['ep']:.1%} "
                              f"escape={st['escape']/st['ep']:.1%} (n={st['ep']})")

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
    ap.add_argument("--ppo", action="store_true", help="use ppo_policy.zip (implies --agent)")
    ap.add_argument("--batch", type=int, default=0, help="batch size shown as the denominator")
    ap.add_argument("--obstacles", type=int, default=0,
                    help="spawn 1..N rough ground obstacles per round (0 = none)")
    args = ap.parse_args()

    if args.dqn and args.ppo:
        sys.exit("Pass only one of --dqn / --ppo.")

    agent_mode = args.agent or args.dqn or args.ppo
    if agent_mode:
        if args.obstacles > 0:
            print("[note] --obstacles is human-mode only for now; the RL env needs "
                  "obstacle perception wired before the agent can see them.")
        kind = "ppo" if args.ppo else "dqn" if args.dqn else "tabular"
        handler = make_agent_handler(kind, args.batch)
        label = f"AGENT/{kind.upper()}"
    else:
        handler = make_human_handler(args.obstacles)
        label = f"HUMAN (obstacles={args.obstacles})"

    async with websockets.serve(handler, "localhost", 8765):
        print(f"Simulation server started on ws://localhost:8765  [{label} mode]")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
