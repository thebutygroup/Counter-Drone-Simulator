# Counter-Drone Simulator

A drone-intercept simulator with a reinforcement-learning interceptor, built in
three escalating stages that share one physics/RL engine.

## Structure

```
Counter-Drone-Simulator/
├── core/                 shared engine (imported by every stage)
│   ├── drone_core.py     physics + the evasive target AI
│   ├── drone_env.py      Gym-style RL environment (reset/step/reward)
│   └── qlearn.py         tabular Q-learning trainer + greedy agent
├── stage1_2d_profile/    STAGE 1 — 2D side-on view (current)
│   ├── index.html        browser viewer (human play / agent / live training)
│   ├── server.py         websocket server: human + --agent modes
│   ├── watch_train.py    train while watching demos in the browser
│   ├── train.py          headless training -> q_policy.npz + curve
│   └── q_policy.npz       trained policy
├── stage2_3d/            STAGE 2 — 3D world (planned)
└── stage3_fpv/           STAGE 3 — first-person + obstacles (planned)
```

The stage apps add `../core` to their import path automatically, so you can run
them straight from their own folder.

## Stage 1 — quick start

From inside `stage1_2d_profile/`:

```bash
pip install -r ../requirements.txt   # numpy, websockets, matplotlib

python server.py                     # human play (arrow keys, R to reset)
python watch_train.py                # train + watch demos; then open index.html
python server.py --agent             # watch the saved policy fly
python train.py --episodes 12000     # headless train (checkpoints as it goes)
python train.py --resume --episodes 4000   # continue from checkpoint.npz
```

### Stored training progress

`train.py` writes four things into the stage folder as it runs:

| File | What it is |
|------|------------|
| `q_policy.npz` | deployable policy — what `server.py --agent` loads |
| `checkpoint.npz` | full training state (policy + episode count + history); resumable, crash-safe (atomic writes) |
| `metrics.csv` | append-only progress log (episode, phase, success/crash rate, reward, ε) — opens in Excel |
| `training_curve.png` | success + reward curves |

Checkpoints are written every `--checkpoint-every` episodes (default 500), so a
Ctrl-C or crash loses at most that many. `--resume` reloads `checkpoint.npz` and
trains more episodes in the hard (EVADE) regime, extending the same history and
metrics log. `q_policy.npz` and `checkpoint.npz` are the same format — either can
be loaded by `server.py --agent`.

For any browser mode, open `index.html` after the server is running. The
**Playback** slider scales demo/agent speed live.

## How the agent works (stage 1)

- **Environment** (`core/drone_env.py`): 7 discrete actions, a 9-D observation
  (geometry + own velocity + own position so it can see the walls), and a dense
  reward = closing bonus − step cost + survival bonus, big bonus on intercept
  (scaled by impact speed), terminal penalty for touching a wall.
- **Agent** (`core/qlearn.py`): tabular Q-learning over a discretised state,
  trained with a 3-phase curriculum (survive → catch → evade).
- **Known ceiling:** tabular Q plateaus on this continuous-control task. The env
  already speaks the Gym API, so a deep agent (DQN/PPO) is a drop-in upgrade and
  is the intended path before stage 2.

## Roadmap

| Stage | View | Physics | Agent perception | Status |
|-------|------|---------|------------------|--------|
| 1 | 2D profile | 2D, gravity | coordinates | done |
| 2 | 3D (three.js) | 6-DOF | coordinates | planned |
| 3 | First-person + obstacles | 6-DOF + collision | camera pixels (vision RL) | planned |

See the README in each stage folder for its specific design notes.
