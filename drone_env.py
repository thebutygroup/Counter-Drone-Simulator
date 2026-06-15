"""
drone_env.py
============
A headless, Gym-style environment for training an interceptor.

API mirrors Gymnasium so you can swap in DQN/PPO later with no env changes:
    obs            = env.reset()
    obs, r, done, info = env.step(action_idx)

Design choices that matter for learning are all collected near the top as
constants so they're easy to tune.

ACTION SPACE (discrete, 7):
    The drone has 4 binary inputs but most combinations are useless (e.g.
    thrust+reverse). We expose a curated set: coast, thrust, reverse, turn
    left/right, and thrust-while-turning (for banked arcs).

OBSERVATION (continuous, 9-vector) -- what the agent "sees":
    [0] dist_norm   distance to target, normalised to ~[0,1]
    [1] rel_angle   bearing to target RELATIVE to heading  [-pi, pi]   (which way to turn)
    [2] angle       absolute heading                       [-pi, pi]   (orientation vs gravity)
    [3] vx          own horizontal velocity
    [4] vy          own vertical velocity                  (am I falling? -> fight gravity)
    [5] tvx         target horizontal velocity             (lead a moving target)
    [6] tvy         target vertical velocity
    [7] px          own x / WIDTH                          (how close to the side walls)
    [8] py          own y / HEIGHT                         (how close to floor/ceiling)

REWARD (dense, potential-based):
    + closing reward  (got nearer this frame)
    - small step cost (be quick)
    - wall penalty    (don't hug the edges)
    + big bonus on intercept (episode ends in success)
"""

import math
import numpy as np

import drone_core
from drone_core import DroneSimulator, WIDTH, HEIGHT, DRONE_RADIUS

DIAG = math.hypot(WIDTH, HEIGHT)

# --- discrete action set: each entry is the input dict for one frame ---------
ACTIONS = [
    {},                                   # 0 coast
    {"thrust": True},                     # 1 thrust forward
    {"reverse": True},                    # 2 reverse / dive
    {"left": True},                       # 3 rotate left
    {"right": True},                      # 4 rotate right
    {"thrust": True, "left": True},       # 5 thrust + bank left
    {"thrust": True, "right": True},      # 6 thrust + bank right
]
N_ACTIONS = len(ACTIONS)

# --- reward shaping knobs ----------------------------------------------------
CLOSING_SCALE = 0.10       # reward per pixel of distance closed this frame
STEP_PENALTY = 0.02        # cost per frame (urgency)
SURVIVAL_BONUS = 0.05      # reward per frame ALIVE -> learn to fly before hunting
INTERCEPT_BONUS = 100.0    # base reward for catching the target
IMPACT_SPEED_BONUS = 3.0   # EXTRA reward per px/frame of CLOSING SPEED at impact
                           #   -> a fast ram scores higher than a gentle drift-in
WALL_DEATH_PENALTY = 25.0  # penalty for touching a wall (episode ends in failure)
WALL_TERMINATES = True     # touching any boundary kills the episode
MAX_STEPS = 600            # episode timeout (~10s at 60fps)


class DroneInterceptEnv:
    def __init__(self, evasion=True, target_speed=drone_core.TARGET_SPEED,
                 wall_terminates=WALL_TERMINATES, seed=None):
        import random
        rng = random.Random(seed)
        self.sim = DroneSimulator(evasion=evasion, target_speed=target_speed, rng=rng)
        self.wall_terminates = wall_terminates
        self._last_frame = None
        self._prev_dist = None
        self._prev_target = None
        self.steps = 0

    # -- core API -------------------------------------------------------------
    def reset(self):
        self.sim.reset()
        self.steps = 0
        self._prev_target = (self.sim.target.x, self.sim.target.y)
        frame = self.sim.step(ACTIONS[0], auto_reset=False)  # one idle frame to seed state
        self._last_frame = frame
        self._prev_dist = frame["state"]["distance"]
        return self._obs(frame)

    def step(self, action_idx):
        prev_tx, prev_ty = self.sim.target.x, self.sim.target.y
        frame = self.sim.step(ACTIONS[action_idx], auto_reset=False)
        self._last_frame = frame
        self.steps += 1

        dist = frame["state"]["distance"]
        # target velocity (finite difference) -- used for the observation and
        # for the closing-speed impact bonus.
        self._tvx = self.sim.target.x - prev_tx
        self._tvy = self.sim.target.y - prev_ty

        # --- reward: dense closing signal, an urgency cost, and a survival
        #     bonus that pays off every frame the drone stays airborne ---
        reward = (self._prev_dist - dist) * CLOSING_SCALE - STEP_PENALTY + SURVIVAL_BONUS

        done = False
        success = False
        crashed = False
        impact_speed = 0.0

        if frame["collision"]:
            # Closing speed at impact = magnitude of player velocity RELATIVE to
            # the target. Faster ram -> bigger bonus.
            p = self.sim.player
            impact_speed = math.hypot(p.vx - self._tvx, p.vy - self._tvy)
            reward += INTERCEPT_BONUS + IMPACT_SPEED_BONUS * impact_speed
            done, success = True, True
        elif self.wall_terminates and self._on_wall():
            # Touching a boundary is now a hard failure.
            reward -= WALL_DEATH_PENALTY
            done, crashed = True, True
        elif self.steps >= MAX_STEPS:
            done = True

        self._prev_dist = dist
        info = {"success": success, "crashed": crashed, "distance": dist,
                "steps": self.steps, "impact_speed": impact_speed}
        return self._obs(frame), reward, done, info

    # -- helpers --------------------------------------------------------------
    def _on_wall(self):
        p = self.sim.player
        e = 1.0
        return (p.x <= DRONE_RADIUS + e or p.x >= WIDTH - DRONE_RADIUS - e or
                p.y <= DRONE_RADIUS + e or p.y >= HEIGHT - DRONE_RADIUS - e)

    def _obs(self, frame):
        p = self.sim.player
        tvx = getattr(self, "_tvx", 0.0)
        tvy = getattr(self, "_tvy", 0.0)
        return np.array([
            frame["state"]["distance"] / DIAG,
            frame["state"]["relative_angle"],
            p.angle,
            p.vx,
            p.vy,
            tvx,
            tvy,
            p.x / WIDTH,    # [7] own x position -> lets the agent SEE the side walls
            p.y / HEIGHT,   # [8] own y position -> lets the agent SEE floor/ceiling
        ], dtype=np.float32)

    def render_frame(self):
        """Frame dict in the exact shape the browser viewer expects."""
        return self._last_frame
