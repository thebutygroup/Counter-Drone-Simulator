# Stage 3 — First-person view + obstacles

**Status: planned.**

The hardest and most interesting stage: a nose-mounted camera, a world with
obstacles to avoid, and an agent that learns to intercept **from what it sees**
rather than from privileged coordinates.

## What changes from stage 2

- **World:** static (and later moving) obstacles the drone must fly around; real
  collision handling against geometry, not just arena walls.
- **Perception — the big one:** the agent's observation becomes the **rendered
  camera image** (a stack of recent frames), not a state vector. This is
  vision-based RL.
- **Agent:** a convolutional policy (CNN encoder → policy/value heads), trained
  with PPO/SAC. Effectively needs a GPU and an offscreen renderer that can run
  many environments in parallel (e.g. headless three.js / a Python 3D engine).

## Why this is a step change, not an increment

Stages 1→2 changed the *physics*. Stage 2→3 changes *how the agent perceives the
world* — coordinates → pixels — which is a different class of problem
(representation learning + control, partial observability, much higher sample
cost). Budget for this accordingly; it is a project in itself.

## Sensible intermediate steps

1. FPV camera for the **human** first (render only), reusing stage-2 physics.
2. Add obstacles + collision to the physics/env, still with coordinate
   observations, and confirm a deep agent can dodge them.
3. Only then switch the observation to pixels and bring in the CNN policy.

## Research / approach — ACTION ITEMS

- [ ] **Evaluate PyBullet (https://pybullet.org) as the 3D physics + render engine.**
      Python-native, fast, headless-friendly, and used widely in RL research — a
      much better fit for training than a browser renderer or a heavy robotics
      stack. Likely the right engine for stages 2–3.
- [ ] **Object segmentation for FPV (the key idea to chase).** PyBullet's
      `getCameraImage()` returns RGB **+ depth + a per-pixel segmentation mask**
      essentially for free. That ground-truth segmentation is valuable for FPV:
      - as an **auxiliary training signal** (predict the mask) to speed up learning
        a useful visual representation,
      - as a **privileged signal** during training that the agent is weaned off
        before deployment,
      - for **reward shaping / debugging** ("is the target actually in view?").
      ACTION: prototype a single-drone + obstacle PyBullet scene, pull RGB + seg
      mask from a nose camera, and confirm the target/obstacles are cleanly
      separable in the mask. Decide how the mask is used (aux task vs privileged).
- [ ] Benchmark PyBullet camera throughput (frames/sec, parallel envs) — vision RL
      is bottlenecked by render speed, so measure before committing.
- [ ] Open question: train on PyBullet seg masks, but plan the sim→real gap early
      if this is ever meant for real hardware.