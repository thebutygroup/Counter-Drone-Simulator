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
IMPACT_SPEED_BONUS = 6.0   # EXTRA reward per px/frame of CLOSING SPEED at impact (was 3.0)
                           #   -> a fast ram scores higher than a gentle drift-in
WALL_DEATH_PENALTY = 10.0  # penalty for touching a wall (was 25.0) -- bolder approaches
WALL_TERMINATES = True     # touching any boundary kills the episode
MAX_STEPS = 600            # episode timeout (~10s at 60fps)

# --- escape terminal: the target reaching the invisible wall ends the round ---
# The target now flees to the wall and is SAFE there. Letting it escape is a
# loss, but a SMALLER one than crashing -- keep ESCAPE_PENALTY < WALL_DEATH_PENALTY
# so the agent never prefers diving into a wall over a near-certain escape.
ESCAPE_PENALTY = 8.0       # penalty when the target reaches safety (episode ends)

# --- A1: impact-speed ramp (top speed is 10 px/frame -> demand a genuinely fast hit) ---
MIN_IMPACT_SPEED = 3.0     # px/frame: below this the speed terms are ~0
FULL_IMPACT_SPEED = 9.0    # px/frame: at/above this the speed terms saturate (~top speed)
# --- A2: dense wall-proximity penalty (in ADDITION to the terminal one) ------
WALL_MARGIN = 100.0        # px from any wall where the dense penalty starts (bigger arena)
WALL_PROX_COEF = 0.12      # per-frame penalty at the wall (was 0.20) -- the target now flees
                           #   TO the wall, so heavy edge-timidity would teach the chaser to
                           #   give up exactly where intercepts happen.
# --- A3: backside-hit orientation --------------------------------------------
IMPACT_ORIENT_BONUS = 15.0 # additive backside gradient (was 40.0) -- reward rear-first harder
BACKSIDE_JACKPOT = 0.0   # BIG multiplicative prize for the HARD combo: fast AND backside-first
                           #   (was 120.0) -- JACKPOT * speed_gate * backside; only a fast
                           #   rear-first hit collects it, slow or prop-first gets ~none.
# --- time-to-intercept: reward quick kills -----------------------------------
TIME_BONUS = 60.0          # bonus for a fast kill, scaled by time_frac below
PAR_STEPS = 150            # "par" time (frames). Instant kill -> full bonus; >= PAR -> none.

# --- obstacles (only active when n_obstacles > 0) ----------------------------
# Perception-AGNOSTIC: computed from obstacle geometry, identical for rays/slots,
# so swapping perception never changes the reward.
OBSTACLE_DEATH_PENALTY = 25.0  # terminal penalty for hitting an obstacle (counts as a crash)
OBSTACLE_MARGIN = 60.0         # px of clearance where the dense penalty starts
OBSTACLE_PROX_COEF = 0.20      # per-frame penalty strength at the obstacle surface


class DroneInterceptEnv:
    def __init__(self, evasion=True, config=None,
                 wall_terminates=WALL_TERMINATES, seed=None,
                 n_obstacles=0, perception="none", perception_kwargs=None,
                 continuous=False):
        import random
        from perception import make_perception
        rng = random.Random(seed)
        self.rng = rng
        self.continuous = continuous
        self.cfg = config or drone_core.cfg
        self.sim = DroneSimulator(
                    config=self.cfg, 
                    evasion=evasion, 
                    rng=rng, 
                    n_obstacles=n_obstacles
                )
        self.wall_terminates = wall_terminates
        self.perception = make_perception(perception, **(perception_kwargs or {}))
        self._last_frame = None
        self._prev_dist = None
        self._prev_target = None
        self.steps = 0

    # -- core API -------------------------------------------------------------
    def reset(self):
        self.sim.reset()
        self.perception.reset(self.rng)
        self.steps = 0
        self._prev_target = (self.sim.target.x, self.sim.target.y)
        idle = np.zeros(2, dtype=np.float32) if self.continuous else ACTIONS[0]
        frame = self.sim.step(idle, auto_reset=False)  # one idle frame to seed state
        self._last_frame = frame
        self._prev_dist = frame["state"]["distance"]
        return self._obs(frame)

    def step(self, action):
        prev_tx, prev_ty = self.sim.target.x, self.sim.target.y
        if self.continuous:
            # Box action: [thrust, turn] in [-1,1]. Clip defensively in case the
            # Gaussian policy emits values outside the box (SB3 doesn't squash).
            act = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
            frame = self.sim.step(act, auto_reset=False)
        else:
            frame = self.sim.step(ACTIONS[int(action)], auto_reset=False)
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
        escaped = False
        impact_speed = 0.0
        backside = 0.0
        time_frac = 0.0

        if frame["collision"]:
            # --- A1: true CLOSING speed = own velocity relative to target,
            #         projected onto the drone->target axis (>=0). ---
            impact_speed = self._impact_speed()
            # --- A3: backside metric. relative_angle ~0 => prop-first (bad),
            #         |relative_angle| ~pi => backside-first (good). ---
            backside = abs(frame["state"]["relative_angle"]) / math.pi   # [0,1]
            # smooth speed ramp (no hard cliff for the optimiser to skirt)
            speed_gate = float(np.clip(
                (impact_speed - MIN_IMPACT_SPEED) /
                (FULL_IMPACT_SPEED - MIN_IMPACT_SPEED), 0.0, 1.0))
            # quick-kill fraction: 1.0 for an instant intercept, 0.0 at/after PAR_STEPS
            time_frac = max(0.0, 1.0 - self.steps / PAR_STEPS)

            reward += INTERCEPT_BONUS                              # base: any hit stays positive
            reward += IMPACT_SPEED_BONUS * impact_speed * speed_gate   # fast (dense gradient)
            reward += IMPACT_ORIENT_BONUS * backside                  # backside (dense gradient)
            reward += BACKSIDE_JACKPOT * speed_gate * backside        # the HARD combo (the real prize)
            reward += TIME_BONUS * time_frac                          # quick kill
            done, success = True, True
        elif frame["escaped"]:
            # Target reached the invisible wall and is safe -> round lost. A
            # bounded loss, smaller than a crash, so the chaser commits to fast
            # intercepts rather than bailing into a wall when an escape looms.
            reward -= ESCAPE_PENALTY
            done, escaped = True, True
        elif self.wall_terminates and self._on_wall():
            # Touching a boundary is still a hard failure.
            reward -= WALL_DEATH_PENALTY
            done, crashed = True, True
        elif frame.get("obstacle_hit"):
            # Hitting a ground obstacle is a crash too (perception-agnostic).
            reward -= OBSTACLE_DEATH_PENALTY
            done, crashed = True, True
        elif self.steps >= MAX_STEPS:
            done = True

        # --- A2: dense wall-proximity penalty applied EVERY step, so the agent
        #         learns to avoid the edge early rather than only at death. ---
        reward += self._wall_penalty()
        # Dense obstacle-proximity penalty (mirrors walls; 0 when no obstacles).
        reward += self._obstacle_penalty()

        self._prev_dist = dist
        info = {"success": success, "crashed": crashed, "escaped": escaped,
                "distance": dist, "steps": self.steps, "impact_speed": impact_speed,
                "backside": backside, "time_frac": time_frac}
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
        base = np.array([
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
        if self.perception.obs_dim():
            return np.concatenate([base, self.perception.observe(self.sim)])
        return base

    # base ego/target observation width (perception is appended after this)
    BASE_OBS_DIM = 9

    def _impact_speed(self):
        """Closing speed at contact: own velocity RELATIVE to the target,
        projected onto the drone->target axis. Clamped >=0 so only genuine
        closing counts (moving fast sideways/away does not earn the bonus).
        Orthogonal to backside: this uses the VELOCITY vector, backside uses
        the HEADING vector -- which is what makes 'fast + backside' learnable."""
        p = self.sim.player
        dx = self.sim.target.x - p.x
        dy = self.sim.target.y - p.y
        d = math.hypot(dx, dy) + 1e-8
        rvx = p.vx - getattr(self, "_tvx", 0.0)   # velocity relative to target
        rvy = p.vy - getattr(self, "_tvy", 0.0)
        return max(0.0, (rvx * dx + rvy * dy) / d)

    def _wall_penalty(self):
        """Dense per-step penalty as the drone enters the edge margin.
        Quadratic: soft at the margin, steep near the wall. The terminal
        WALL_DEATH_PENALTY still fires on contact -- this is in addition."""
        p = self.sim.player
        d = min(p.x, WIDTH - p.x, p.y, HEIGHT - p.y)
        if d >= WALL_MARGIN:
            return 0.0
        t = 1.0 - d / WALL_MARGIN
        return -WALL_PROX_COEF * t * t

    def _obstacle_penalty(self):
        """Dense per-step penalty as the drone enters an obstacle's margin.
        Perception-agnostic (geometry only). 0 when no obstacles are present."""
        if not self.sim.obstacles:
            return 0.0
        import objects
        p = self.sim.player
        nearest = min(objects.obstacle_clearance(o, p.x, p.y) for o in self.sim.obstacles)
        d = nearest - DRONE_RADIUS          # clearance from the drone's rim
        if d >= OBSTACLE_MARGIN:
            return 0.0
        t = 1.0 - max(d, 0.0) / OBSTACLE_MARGIN
        return -OBSTACLE_PROX_COEF * t * t

    def render_frame(self):
        """Frame dict in the exact shape the browser viewer expects."""
        return self._last_frame
