# Counter-Drone Simulator

## **NOTE: This repo has been abandon and gone private as of June 16th 2026**

A drone-intercept simulator with reinforcement-learning agents, built on one shared
physics/RL engine. The project escalates along **two axes**: agent capability
(single-agent → adversarial self-play → multi-agent team fight) and world fidelity
(2D → 3D → first-person vision). **Capability is developed first, in fast 2D**, and only
the matured game is ported up the fidelity axis.

## Current direction (the through-line)

Stage 1 single-agent interception works: tabular Q-learning plateaus on this
continuous-control task, and a neural **DQN clearly breaks that plateau**. That makes the
2D engine a proven, fast substrate — so the next iterations stay in 2D and grow the
*agents*, not the renderer. The planned sequence:

| Step | Goal | Algorithm | Status |
|---|---|---|---|
| **A. Sharpen the interceptor** | fly faster into the target, crash into walls less, hit backside-first (away from props) | DQN → **PPO** | in progress |
| **B. Smarter evasion via self-play** | make the evader a *learned* agent maximising survival; interceptor co-trains against it | PPO self-play | next |
| **C. Counter-counter-drone team fight** | add a counter-interceptor role → 3-way game (evader / interceptor / counter) | multi-agent PPO (PettingZoo) | the big one |
| **2D → 3D port** | move the *matured multi-agent game* to 6-DOF physics | PPO/SAC | after C |
| **3D → FPV** | swap coordinate observations for nose-cam pixels | CNN + PPO | last |

**Design principle: do multi-agent in 2D before 3D.** MARL is the hard part
(non-stationary dynamics, unstable self-play, reward/curriculum tuning); rendering isn't.
Prototyping the team fight in the existing fast engine keeps iteration cheap and isolates
the genuinely difficult problem from the rendering port.

## Structure

```
Counter-Drone-Simulator/
├── core/                 shared engine (imported by every stage)
│   ├── drone_core.py     2D physics + the evasive target AI
│   ├── drone_env.py      single-agent Gym-style env (reset/step/reward)
│   ├── gym_env.py        Gymnasium adapter for Stable-Baselines3 (DQN/PPO)
│   ├── qlearn.py         tabular Q-learning trainer + greedy agent (baseline)
│   └── marl_env.py       (PLANNED) PettingZoo multi-agent env for steps B/C
├── stage1_2d_profile/    STAGE 1 — 2D side-on view (current home of all RL work)
│   ├── index.html        browser viewer (human / agent / live-training, speed slider, HUD)
│   ├── server.py         websocket server: human | --agent (tabular) | --agent --dqn
│   ├── train.py          headless tabular training (checkpoint/resume/metrics)
│   ├── watch_train.py    train tabular while watching in the browser
│   ├── train_dqn.py      DQN (Stable-Baselines3) trainer
│   ├── train_ppo.py      (PLANNED) PPO trainer — go-forward algorithm for A/B/C
│   ├── benchmark.py      tabular-vs-DQN compare on identical seeds → CSV
│   ├── dqn_policy.zip     trained neural policy
│   └── q_policy.npz       trained tabular policy (baseline)
├── stage2_3d/            STAGE 2 — 3D world (planned; port of the matured 2D game)
└── stage3_fpv/           STAGE 3 — first-person vision + obstacles (planned)
```

The stage apps add `../core` to their import path automatically — run them from inside
their own folder. `core/` is a library and is never run directly.

## Stage 1 — quick start

From inside `stage1_2d_profile/`:

```bash
pip install -r ../requirements.txt          # numpy, websockets, matplotlib (tabular path + viewer)
pip install -r requirements-dqn.txt          # stable-baselines3[extra] -> torch, gymnasium (neural path)

# --- play / watch ---
python server.py                             # human play (arrow keys, R to reset) -> open index.html
python server.py --agent                     # watch the saved TABULAR policy
python server.py --agent --dqn               # watch the saved NEURAL policy

# --- train ---
python train_dqn.py --timesteps 600000       # neural; writes dqn_policy.zip (USE THIS for a real model)
python train_dqn.py --resume                 # continue from dqn_policy.zip
python train.py --episodes 12000             # tabular baseline (checkpoints as it goes)
python watch_train.py                        # train tabular + watch demos live

# --- score ---
python benchmark.py --episodes 500 --label dqn-600k   # appends to benchmark_results.csv
```

See `stage1_2d_profile/README.md` for the full per-script reference, the training
curriculum, the reward knobs, and the path to a fully-trained model.

### Stored training progress

The tabular trainers write into the stage folder as they run:

| File | What it is |
|------|------------|
| `q_policy.npz` | deployable tabular policy — what `server.py --agent` loads |
| `checkpoint.npz` | full training state (policy + episode count + history); resumable, crash-safe (atomic writes) |
| `metrics.csv` | append-only progress log (episode, phase, success/crash rate, reward, ε) |
| `training_curve.png` | success + reward curves |

DQN writes `dqn_policy.zip` (final, saved at the end of every curriculum phase) and
`dqn_checkpoints/` (every 25k steps, resumable). For any browser mode, start the server
first, then open `index.html`; the **Playback** slider scales speed live.

## How the agent works (stage 1)

- **Physics** (`core/drone_core.py`): side-on, rotating-thrust model — Up = thrust along
  heading, Down = reverse, Left/Right = rotate. Gravity, drag, turn rate as constants.
  Touching any wall is terminal. Intercept = distance < 2·`DRONE_RADIUS`. The target
  wanders and flees radially when the player is within 200px (currently exploitable —
  step B fixes this by making the evader learn).
- **Environment** (`core/drone_env.py`): 7 discrete actions, 9-D observation (geometry +
  own velocity + target velocity + own position so it can see the walls). Reward is dense
  and being sharpened for step A:
  - closing bonus − step cost + survival bonus (base shaping),
  - **A1** impact-speed bonus on a smooth ramp (reward fast *closing* hits),
  - **A2** dense wall-proximity penalty inside an edge margin (avoid the wall *early*, not
    just at death — terminal penalty still applies),
  - **A3** backside-hit bonus (`backside = |relative_angle|/π`) — reward hitting with the
    rear, away from the props; a gate-mode jackpot for *fast + backside-first* is the
    curriculum's hard target.
- **Agents:** tabular Q-learning (`core/qlearn.py`) is the baseline and has plateaued.
  **DQN (SB3) is the working neural agent.** **PPO is the go-forward algorithm** — A3's
  precise reverse-thrust timing wants it, and B/C (self-play + multi-agent) need it
  regardless, so the switch happens during step A to tune the reward once rather than twice.

## Foundation notes for the next iterations

These are the commitments that keep A → B → C → 3D coherent:

1. **The env interface is the contract.** `gym_env.py` already isolates SB3 from the raw
   env; reward edits in `drone_env.py` flow through automatically. The multi-agent jump
   (step C) introduces `core/marl_env.py` on the **PettingZoo Parallel API** rather than
   bending the single-agent env — per-agent obs/action/reward, with observations that
   scale with entity count.
2. **Per-role rewards, not one shared reward.** Interceptor: +hit evader, −hit by counter,
   −time, −wall. Counter: +hit interceptor, −time, −wall. Evader: +survival, −proximity.
   Designing rewards per role now avoids a rewrite later.
3. **PPO is the standard learner across A/B/C** — independent learners per role to start,
   then self-play for the adversarial pairs, then heavier centralised-critic MARL only if
   independent PPO stalls.
4. **Benchmark every behaviour change.** `benchmark.py` extends with impact-speed and
   backside-rate columns as those rewards land; "trained" means the metric plateaued, not
   that timesteps ran out.
5. **Port last.** Only after the multi-agent game matures in 2D does it move to PyBullet
   (3D), then to nose-cam pixels (FPV).

## Roadmap

| Stage / step | Axis | What changes | Status |
|---|---|---|---|
| 1 — single-agent 2D | capability | tabular baseline; DQN breaks the plateau | done |
| A — sharpen interceptor | capability | reward shaping (fast / no-crash / backside); DQN → PPO | in progress |
| B — learned evader | capability | self-play; evader becomes an agent | next |
| C — team fight | capability | counter-interceptor role; PettingZoo multi-agent PPO | planned |
| 2 — 3D | fidelity | 6-DOF physics (PyBullet); obs/action grow | planned |
| 3 — FPV | fidelity | camera-pixel observations; CNN policy | planned |

See the README in each stage folder for that stage's design notes.
