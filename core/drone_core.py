"""
drone_core.py
=============
Single source of truth for the simulation physics.

Both the live game (server.py) and the reinforcement-learning environment
(drone_env.py) import from here, so the agent trains on EXACTLY the dynamics
the game renders. If you retune a constant, do it here and both stay in sync.
"""

import math
import random

WIDTH, HEIGHT = 2400, 1800      # expanded arena (was 800x600)
DRONE_RADIUS = 30
FPS = 60                       # frames per second the sim is paced at

# ---------------------------------------------------------------------------
# Metric scale: ONE source of truth for real-world units. Objects are defined
# in metres (see objects.py) and converted to pixels via this. Phase 2 (PyBullet)
# is metre-native, so this is the bridge that keeps 2D and 3D dimensions in sync.
#   arena   = 48 m x 36 m
#   drone r = 1.2 m ;  TOP_SPEED 10 px/frame ~ 24 m/s ;  gravity ~ 8.6 m/s^2
# ---------------------------------------------------------------------------
PIXELS_PER_METER = 25.0

# ---------------------------------------------------------------------------
# Speed limits
# ---------------------------------------------------------------------------
TOP_SPEED = 10.0               # chase drone HARD velocity cap (px/frame)
TARGET_SPEED_FRAC = 0.75       # target tops out at this fraction of the chaser's cap
TARGET_SPEED = TOP_SPEED * TARGET_SPEED_FRAC   # 7.5 px/frame, constant -- no inertia

# ---------------------------------------------------------------------------
# Spawn rules: both drones start away from the walls, and no further apart than
# the chaser could cover in ~1 second of top-speed flight.
# ---------------------------------------------------------------------------
SPAWN_WALL_MARGIN = 120        # both drones start >= this many px from any wall
SPAWN_MAX_SEP = TOP_SPEED * FPS    # 600px: reachable in ~1s at top speed
SPAWN_MIN_SEP = 200            # not trivially on top of each other

# ---------------------------------------------------------------------------
# Target evasion AI -- tuning knobs (see comments in DroneSimulator.step)
# ---------------------------------------------------------------------------
EVASION_TRIGGER_RADIUS = 200   # player within this distance (px) spooks the target
ESCAPE_LOOKAHEAD = 300         # legacy: fallback flee distance if the boundary ray degenerates
WAYPOINT_REACHED_DIST = 20     # distance (px) at which a waypoint counts as "reached"

# ---------------------------------------------------------------------------
# Invisible "escape" wall. A rectangle inset from the true arena edge by
# WALL_OFFSET. The target has ESCAPED the moment its CENTRE crosses this line
# (comes within WALL_OFFSET of any real edge). Offset = one drone width so the
# target clears the line well before its centre would hit the hard clamp at
# DRONE_RADIUS -- it always has room to actually reach safety.
# Single source of truth read by: the flee AI (below), the RL escape terminal
# (drone_env.py) and the browser debug overlay (index.html).
# ---------------------------------------------------------------------------
WALL_OFFSET = 2 * DRONE_RADIUS   # 60 px = drone width; invisible wall inset from each edge


def _ray_to_boundary(x, y, ang, margin):
    """Cast a ray from (x, y) along `ang` to the rectangle inset from the arena
    by `margin`. Used by the target's flee logic so it commits to a REAL wall
    (runs all the way to the edge) instead of stopping at a fixed look-ahead.
    Falls back to ESCAPE_LOOKAHEAD if the ray is degenerate."""
    cx, cy = math.cos(ang), math.sin(ang)
    lo_x, hi_x = margin, WIDTH - margin
    lo_y, hi_y = margin, HEIGHT - margin
    best_t = None
    eps = 1e-9
    if cx > eps:
        best_t = (hi_x - x) / cx
    elif cx < -eps:
        best_t = (lo_x - x) / cx
    if abs(cy) > eps:
        ty = (hi_y - y) / cy if cy > 0 else (lo_y - y) / cy
        if ty > 0 and (best_t is None or ty < best_t):
            best_t = ty
    if best_t is None or best_t <= 0:
        best_t = ESCAPE_LOOKAHEAD
    hx = x + cx * best_t
    hy = y + cy * best_t
    return (min(max(hx, lo_x), hi_x), min(max(hy, lo_y), hi_y))


class Drone:
    def __init__(self, x, y, angle=-math.pi / 2):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = angle

        # Physics constants (profile view)
        self.thrust_power = 0.45
        self.turn_speed = 0.10   # a touch nimbler -> easier to dodge walls
        self.drag = 0.99
        self.gravity = 0.06      # gentler -> survivable to hover (still a real pull)
        self.top_speed = TOP_SPEED   # hard velocity-magnitude cap

    def apply_physics(self, thrust=False, reverse=False, left=False, right=False):
        """DISCRETE control (legacy 7-action set: live game, tabular, DQN).
        Booleans: left/right rotate by -/+ turn_speed; thrust/reverse push
        +/-1 * thrust_power along heading. Kept BIT-FOR-BIT equivalent to the
        original so it remains a clean comparison baseline against continuous.
        (thrust & reverse both set cancel, as before; both clear = coast.)"""
        turn = (1.0 if right else 0.0) - (1.0 if left else 0.0)
        signed_thrust = (1.0 if thrust else 0.0) - (1.0 if reverse else 0.0)
        self._apply(signed_thrust, turn)

    def apply_physics_continuous(self, thrust, turn):
        """CONTINUOUS control (2-D Box action -- the lever-1 upgrade).
            thrust in [-1, 1]  signed magnitude along heading (+fwd / -reverse)
            turn   in [-1, 1]  signed rotate rate (+right / -left)
        Unlike the discrete set this lets the agent reverse-thrust AND rotate in
        the SAME frame, and modulate thrust magnitude -- the exact freedoms the
        'arrive fast, rear-first' maneuver needs. Inputs are clamped to [-1,1]
        so an unsquashed Gaussian policy can't exceed the actuator limits."""
        t = max(-1.0, min(1.0, float(thrust)))
        r = max(-1.0, min(1.0, float(turn)))
        self._apply(t, r)

    def _apply(self, signed_thrust, turn):
        """Shared dynamics core. signed_thrust, turn are floats in [-1, 1].
        Ordering matches the original discrete model exactly: rotate first, then
        thrust along the NEW heading, then drag -> gravity -> speed clamp ->
        integrate -> walls. Discrete and continuous therefore share one integrator
        and stay in sync if a physics constant is retuned."""
        # 1. Orientation (rotate first, so thrust applies along the new heading)
        self.angle += turn * self.turn_speed

        # 2. Thrust along heading (sign picks fwd/reverse, magnitude scales it)
        if signed_thrust:
            self.vx += math.cos(self.angle) * self.thrust_power * signed_thrust
            self.vy += math.sin(self.angle) * self.thrust_power * signed_thrust

        # 3. Drag on the drone's own motion (applied before gravity)
        self.vx *= self.drag
        self.vy *= self.drag

        # 4. Gravity = constant downward acceleration
        self.vy += self.gravity

        # 4b. Top-speed clamp: cap velocity MAGNITUDE (preserves heading).
        speed = math.hypot(self.vx, self.vy)
        if speed > self.top_speed:
            scale = self.top_speed / speed
            self.vx *= scale
            self.vy *= scale

        # 5. Integrate position
        self.x += self.vx
        self.y += self.vy

        # 6. Boundary collisions
        if self.x < DRONE_RADIUS:
            self.x, self.vx = DRONE_RADIUS, 0
        if self.x > WIDTH - DRONE_RADIUS:
            self.x, self.vx = WIDTH - DRONE_RADIUS, 0
        if self.y < DRONE_RADIUS:
            self.y, self.vy = DRONE_RADIUS, 0
        if self.y > HEIGHT - DRONE_RADIUS:
            self.y, self.vy = HEIGHT - DRONE_RADIUS, 0


class DroneSimulator:
    def __init__(self, evasion=True, target_speed=TARGET_SPEED, rng=None, n_obstacles=0):
        # `evasion` / `target_speed` let the RL side dial difficulty (curriculum).
        # `rng` lets training seed reproducibly without touching global random.
        # `n_obstacles` (0 = none) spawns 1..N rough ground obstacles per episode.
        self.evasion = evasion
        self.target_speed = target_speed
        self.rng = rng or random.Random()
        self.n_obstacles = n_obstacles
        self.obstacles = []
        self.reset()

    def _spawn_point(self):
        """A point at least SPAWN_WALL_MARGIN from every wall."""
        m = SPAWN_WALL_MARGIN
        return (self.rng.uniform(m, WIDTH - m), self.rng.uniform(m, HEIGHT - m))

    def _spawn_near(self, px, py):
        """A point [SPAWN_MIN_SEP, SPAWN_MAX_SEP] from (px,py) and inside the
        wall margin. SPAWN_MAX_SEP is one second of top-speed flight, so the
        target always starts reachable. Falls back to a clamped point if no
        valid angle is found (corner spawns)."""
        m = SPAWN_WALL_MARGIN
        tx = ty = None
        for _ in range(100):
            sep = self.rng.uniform(SPAWN_MIN_SEP, SPAWN_MAX_SEP)
            ang = self.rng.uniform(0, 2 * math.pi)
            tx = px + math.cos(ang) * sep
            ty = py + math.sin(ang) * sep
            if m <= tx <= WIDTH - m and m <= ty <= HEIGHT - m:
                return tx, ty
        return (min(max(tx, m), WIDTH - m), min(max(ty, m), HEIGHT - m))

    def reset(self):
        # Player spawns away from the walls (no more fixed corner start).
        px, py = self._spawn_point()
        self.player = Drone(px, py)

        # Target spawns within ~1s of top-speed flight, also away from walls.
        tx, ty = self._spawn_near(px, py)
        self.target = Drone(tx, ty)
        self.target.top_speed = self.target_speed   # cosmetic; target moves directly

        # First wander waypoint somewhere legal.
        self.target_wp_x, self.target_wp_y = self._spawn_point()

        # Obstacles: 1..n_obstacles rough ground objects, placed clear of both
        # drone spawns. Gated on n_obstacles so the RL env is unaffected at 0.
        if self.n_obstacles > 0:
            import objects
            n = self.rng.randint(1, self.n_obstacles)
            avoid = [(self.player.x, self.player.y, DRONE_RADIUS),
                     (self.target.x, self.target.y, DRONE_RADIUS)]
            self.obstacles = objects.spawn_obstacles(self.rng, n, avoid)
        else:
            self.obstacles = []

    def step(self, actions, auto_reset=True):
        """Advance one frame.

        actions:    dict with bool keys thrust/reverse/left/right
        auto_reset: True for the live game (snap to a new round on intercept).
                    The RL env passes False so it can end the episode itself.
        """
        # 1. Player physics. `actions` is EITHER a discrete input dict
        #    (live game / tabular / DQN) OR a 2-vector [thrust, turn] in [-1,1]
        #    (continuous PPO). Dispatch on type so the live game path is untouched.
        if isinstance(actions, dict):
            self.player.apply_physics(
                thrust=actions.get('thrust', False),
                reverse=actions.get('reverse', False),
                left=actions.get('left', False),
                right=actions.get('right', False),
            )
        else:
            self.player.apply_physics_continuous(actions[0], actions[1])

        # =================================================================
        # 2. TARGET EVASION AI  (A) FLEE  (B) WANDER  (C) MOVE
        # =================================================================
        dist_to_player = math.hypot(self.player.x - self.target.x,
                                    self.player.y - self.target.y)

        # (A) FLEE: if threatened, plot an escape waypoint directly away.
        # (A) FLEE: if threatened, RUN FOR THE WALL. Pick the escape direction
        #     (directly away from the player) and project it onto the physical
        #     boundary, so the waypoint sits ON a real edge. Once the target's
        #     centre crosses the invisible WALL_OFFSET line it has ESCAPED (round
        #     over in its favour), so it commits fully to the edge rather than
        #     hovering a fixed look-ahead out.
        if self.evasion and dist_to_player < EVASION_TRIGGER_RADIUS:
            escape_angle = math.atan2(self.target.y - self.player.y,
                                      self.target.x - self.player.x)
            self.target_wp_x, self.target_wp_y = _ray_to_boundary(
                self.target.x, self.target.y, escape_angle, DRONE_RADIUS)

        # (B) WANDER: reached the waypoint? pick a fresh random one.
        dist_to_wp = math.hypot(self.target_wp_x - self.target.x,
                                self.target_wp_y - self.target.y)
        if dist_to_wp < WAYPOINT_REACHED_DIST:
            self.target_wp_x, self.target_wp_y = self._spawn_point()

        # (C) MOVE: one constant-speed step toward the waypoint.
        wp_angle = math.atan2(self.target_wp_y - self.target.y,
                              self.target_wp_x - self.target.x)
        self.target.x += math.cos(wp_angle) * self.target_speed
        self.target.y += math.sin(wp_angle) * self.target_speed

        # 3. State data (heading-relative bearing to target, wrapped to [-pi, pi])
        distance = dist_to_player
        target_angle = math.atan2(self.target.y - self.player.y,
                                  self.target.x - self.player.x)
        relative_angle = (target_angle - self.player.angle + math.pi) % (2 * math.pi) - math.pi

        # 4. Intercept check
        is_collision = distance < (DRONE_RADIUS * 2)

        # 4b. ESCAPE check: target reached safety if its centre crossed the
        #     invisible wall. Only counts when the target is actually EVADING --
        #     a docile (evasion-off) wanderer drifting into the edge is not an
        #     "escape", so early curriculum phases aren't lost to random drift.
        target_escaped = self.evasion and (
            self.target.x <= WALL_OFFSET or
            self.target.x >= WIDTH - WALL_OFFSET or
            self.target.y <= WALL_OFFSET or
            self.target.y >= HEIGHT - WALL_OFFSET)

        # Live game: snap to a fresh round on either terminal outcome. The RL
        # env passes auto_reset=False and ends the episode itself.
        if (is_collision or target_escaped) and auto_reset:
            self.reset()

        # 4c. Obstacle hit check (player only; the scripted target ignores them).
        obstacle_hit = False
        if self.obstacles:
            import objects
            for obs in self.obstacles:
                if objects.obstacle_hit(obs, self.player.x, self.player.y, DRONE_RADIUS):
                    obstacle_hit = True
                    break

        return {
            "drone": {"x": self.player.x, "y": self.player.y, "angle": self.player.angle},
            "target": {"x": self.target.x, "y": self.target.y},
            "state": {"distance": distance, "relative_angle": relative_angle},
            "collision": is_collision,
            "escaped": target_escaped,
            "obstacle_hit": obstacle_hit,
            "obstacles": [{"color": o["color"], "render": o["render"]} for o in self.obstacles],
            "arena": {"w": WIDTH, "h": HEIGHT, "wall_offset": WALL_OFFSET},
        }