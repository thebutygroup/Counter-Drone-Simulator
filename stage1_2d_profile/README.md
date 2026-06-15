# Stage 1 — 2D Profile Interceptor

Single-agent counter-drone interceptor trained on the shared 2D engine in `../core`.
Side-on view, rotating-thrust physics, evasive target. Two learners share one env:
a tabular Q-learning baseline and a neural DQN (Stable-Baselines3).

## Golden rule

**Run every command from inside `stage1_2d_profile/`.** `../core` is a library and is
never run directly — the scripts add it to `sys.path` themselves, but only if the cwd
is this folder.

```bash
cd stage1_2d_profile
```

---

## Prerequisites

Python 3.10+ and:

```bash
pip install numpy websockets matplotlib        # tabular path + viewer + curves
pip install "stable-baselines3[extra]"         # neural path: pulls torch, gymnasium, tensorboard
```

The tabular path (`train.py`, `watch_train.py`) needs only the first line.
DQN (`train_dqn.py`, and `server.py --dqn`) needs SB3/torch.

---

## TL;DR — zero to a deployable neural model

```bash
cd stage1_2d_profile
python train_dqn.py --timesteps 600000     # ~smoke/real run; writes dqn_policy.zip
python server.py --agent --dqn             # serve the trained policy on ws://localhost:8765
# open index.html in a browser -> watch it intercept
python benchmark.py --label dqn-600k       # score it vs the tabular baseline
```

`train_dqn.py` runs a 3-phase curriculum and saves `dqn_policy.zip` at the end of
**every** phase, so even an interrupted run leaves you a usable model.

---

## Scripts

| Script | What it does | Key flags | Produces |
|---|---|---|---|
| `train.py` | Tabular Q-learning, headless, with curriculum + checkpoint/resume | `--episodes 12000` `--seed 0` `--resume` `--checkpoint-every 500` | `q_policy.npz`, `checkpoint.npz`, `metrics.csv`, `training_curve.png` |
| `train_dqn.py` | DQN (SB3) on the same env. **Use this for a real model.** | `--timesteps 1500000` `--seed 0` `--no-curriculum` `--resume` | `dqn_policy.zip`, `dqn_checkpoints/` |
| `watch_train.py` | Train the **tabular** agent while watching live in the browser | `--fresh` (else resumes `checkpoint.npz`) | same as `train.py` + live stream |
| `server.py` | Serve a trained policy to `index.html` over websocket | `--agent` (tabular) · `--agent --dqn` (neural) · `--batch N` (HUD denominator) · none = human play | — |
| `benchmark.py` | Score tabular vs DQN on identical seeds, append to CSV | `--episodes 500` `--label "dqn-600k"` | `benchmark_results.csv` |
| `index.html` | Browser viewer (human / agent / live-training, speed slider, HUD) | — | — |

Default `train_dqn.py` is **1.5M** timesteps; use `--timesteps 600000` for a faster
first pass.

---

## The curriculum (why phased training matters)

`train_dqn.py` trains one model across three phases, swapping the env between them
(survival is learned before hunting, hunting before evasion):

| Phase | Fraction of timesteps | Evasion | `closing_scale` | Goal |
|---|---|---|---|---|
| `survive` | 0.25 | off | 0.0 | stay airborne, don't crash |
| `catch` | 0.35 | off | 0.10 | intercept a non-evading target |
| `evade` | 0.40 | on | 0.10 | intercept the evading target |

`--no-curriculum` skips straight to a single `evade` phase (harder, slower to learn —
use only for ablations). The tabular `train.py` runs an analogous curriculum internally.

---

## Recommended path to a "fully trained" model

1. **Smoke test (10–15 min).** Confirm nothing throws and reward isn't NaN:
   ```bash
   python train_dqn.py --timesteps 600000
   ```
   These trainers were written without torch in-house, so treat the first launch as a
   smoke test and report any SB3/gymnasium API tracebacks.

2. **Full run.** Once the smoke run is clean:
   ```bash
   python train_dqn.py --timesteps 1500000 --seed 0
   ```
   Checkpoints land in `dqn_checkpoints/` every 25k steps (crash-safe / resumable).

3. **Resume if interrupted:**
   ```bash
   python train_dqn.py --resume --timesteps 1500000
   ```
   (continues from `dqn_policy.zip`).

4. **Watch it:**
   ```bash
   python server.py --agent --dqn
   ```
   then open `index.html`. The terminal prints the mode and `ws://localhost:8765`.

5. **Benchmark before you call it done:**
   ```bash
   python benchmark.py --episodes 500 --label dqn-1.5M
   ```
   A model is "fully trained" when success rate has plateaued **and** crash rate is low
   on the EVADE regime — not just when timesteps ran out. Compare `--label`led runs in
   `benchmark_results.csv` to confirm the curve flattened.

`benchmark_results.csv` columns:
`timestamp, agent, label, regime, episodes, success_rate, crash_rate, timeout_rate,
avg_catch_frames, avg_impact_speed, params, disk_kb, infer_us, decisions_per_sec,
train_size, train_unit, success_per_100k_params`.

---

## Reward shaping (current terms — edit in `../core/drone_env.py`)

All knobs are constants at the top of `drone_env.py`. A reward edit is picked up
automatically on the next training launch (the SB3 adapter passes reward straight
through) — but it does **not** retro-apply to an already-saved policy, so retrain after
changing reward.

| Knob | Effect |
|---|---|
| `CLOSING_SCALE` | dense reward per pixel closed (set per-phase by the curriculum) |
| `STEP_PENALTY` / `SURVIVAL_BONUS` | urgency vs. learn-to-fly-first balance |
| `INTERCEPT_BONUS` | base reward for any hit |
| `IMPACT_SPEED_BONUS` + `MIN_IMPACT_SPEED` / `FULL_IMPACT_SPEED` | **A1** — reward fast closing hits via a smooth speed ramp |
| `WALL_MARGIN` / `WALL_PROX_COEF` | **A2** — dense per-step penalty inside the edge margin (terminal `WALL_DEATH_PENALTY` still applies) |
| `IMPACT_ORIENT_BONUS` | **A3** — reward backside-first hits (`backside = |relative_angle|/pi`) |
| `USE_BACKSIDE_GATE` | **A3 mode.** `False` = additive (train this first). `True` = multiplicative jackpot for fast + backside-first hits (curriculum step 2) |

**Order:** train with `USE_BACKSIDE_GATE = False` first (dense signal), confirm the agent
is fast and not suiciding, then flip to `True` and continue-train. A gate from scratch
starves early learning. The combo "fast + backside-first" likely needs PPO rather than
DQN — see the project handoff.

> A2 caveat: a wall-averse agent can start refusing kills in corners. Watch corner-
> intercept rate; if it tanks, drop `WALL_PROX_COEF` (~0.08) or make the penalty
> directional (only when velocity points into the wall).

---

## Artifacts produced

| File | By | Notes |
|---|---|---|
| `dqn_policy.zip` | `train_dqn.py` | the deployable neural policy (`DQN.load`) |
| `dqn_checkpoints/` | `train_dqn.py` | periodic, resumable |
| `q_policy.npz` | `train.py` / `watch_train.py` | deployable tabular policy |
| `checkpoint.npz` | tabular trainers | resume point |
| `metrics.csv` / `training_curve.png` | tabular trainers | learning curve |
| `benchmark_results.csv` | `benchmark.py` | one row per labelled run, appended |

---

## Troubleshooting

- **`No checkpoint found ... run without --resume first`** — you passed `--resume` with no
  prior run; drop the flag for the first run.
- **Browser shows nothing** — the server must be running first; `index.html` connects to
  `ws://localhost:8765`. Start `server.py` (or `watch_train.py`), then open the page.
- **`ModuleNotFoundError: drone_core` / `gym_env`** — you're not in `stage1_2d_profile/`.
- **SB3 import error** — `pip install "stable-baselines3[extra]"`; the tabular path
  doesn't need it.