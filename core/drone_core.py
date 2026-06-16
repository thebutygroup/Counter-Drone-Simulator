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
from dataclasses import dataclass
from typing import Dict

WIDTH, HEIGHT = 1800, 1350      # expanded arena (was 800x600)
DRONE_RADIUS = 10
FPS = 60                        # frames per second the sim is paced at

# ---------------------------------------------------------------------------
# Metric scale: ONE source of truth for real-world units. Objects are defined
# in metres (see objects.py) and converted to pixels via this. Phase 2 (PyBullet)
# is metre-native, so this is the bridge that keeps 2D and 3D dimensions in sync.
#   arena   = 48 m x 36 m
#   drone r = 1.2 m ;  TOP_SPEED 10 px/frame ~ 24 m/s ;  gravity ~ 8.6 m/s^2
# ---------------------------------------------------------------------------
PIXELS_PER_METER = 25.0

# ---------------------------------------------------------------------------
# Spawn rules
# ---------------------------------------------------------------------------
SPAWN_WALL_MARGIN = 400        # both drones start >= this many px from any wall
SPAWN_MAX_SEP = 1000    # 600px: reachable in ~1s at top speed
SPAWN_MIN_SEP = 400            # not trivially on top of each other

# ---------------------------------------------------------------------------
# Target evasion AI -- tuning knobs
# ---------------------------------------------------------------------------
EVASION_TRIGGER_RADIUS = 200   # player within this distance (px) spooks the target
ESCAPE_LOOKAHEAD = 300         # legacy: fallback flee distance if the boundary ray degenerates
WAYPOINT_REACHED_DIST = 20     # distance (px) at which a waypoint counts as "reached"

# ---------------------------------------------------------------------------
# Invisible "escape" wall.
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

@dataclass(frozen=True)
class SimulationConfig:
    # Physical Constants
    width: int = 1800
    height: int = 1350
    drone_radius: int = 30
    fps: int = 60
    pixels_per_meter: float = 25.0
    
    # Speed/Control Limits
    top_speed: float = 45.0
    thrust_power: float = 4
    turn_speed: float = 0.25
    reverse_thrust_frac: float = 0.5
    max_reverse_speed_frac = .85 
    
    # Target Ratios
    target_speed_frac: float = 0.25
    target_thrust_frac: float = 0.4
    target_turn_frac: float = 0.4

    # Spawn/Evasion settings
    spawn_wall_margin: int = 400
    evasion_trigger_radius: int = 200
    escape_lookahead: int = 300
    waypoint_reached_dist: int = 20
    
    @property
    def wall_offset(self) -> int:
        return 2 * self.drone_radius

    @property
    def target_stats(self) -> Dict[str, float]:
        return {
            'target_speed': self.top_speed * self.target_speed_frac,
            'thrust_power': self.thrust_power * self.target_thrust_frac,
            'turn_speed': self.turn_speed * self.target_turn_frac
        }

    @property
    def spawn_max_sep(self) -> float:
        return self.top_speed * self.fps

# Instantiate the "Single Source of Truth"
cfg = SimulationConfig()


class Drone:
    def __init__(self, x, y, config: SimulationConfig, angle=-math.pi / 2):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = angle

        # Physics constants driven ENTIRELY by the config object
        self.thrust_power = config.thrust_power
        self.turn_speed = config.turn_speed
        self.top_speed = config.top_speed
        self.reverse_thrust_frac = config.reverse_thrust_frac

        self.drag = 0.99
        self.gravity = 0.06      # gentler -> survivable to hover (still a real pull)

    def apply_physics(self, thrust=False, reverse=False, left=False, right=False):
        """DISCRETE control (legacy 7-action set: live game, tabular, DQN)."""
        turn = (1.0 if right else 0.0) - (1.0 if left else 0.0)
        signed_thrust = (1.0 if thrust else 0.0) - (1.0 if reverse else 0.0)
        self._apply(signed_thrust, turn)

    def apply_physics_continuous(self, thrust, turn):
        """CONTINUOUS control (2-D Box action -- the lever-1 upgrade)."""
        t = max(-1.0, min(1.0, float(thrust)))
        r = max(-1.0, min(1.0, float(turn)))
        self._apply(t, r)

    def _apply(self, signed_thrust, turn):
        """Shared dynamics core."""
        # 1. Orientation
        self.angle += turn * self.turn_speed

        # 2. Thrust (reverse is weaker by reverse_thrust_frac)
        if signed_thrust:
            power = self.thrust_power
            if signed_thrust < 0:
                power *= self.reverse_thrust_frac
            self.vx += math.cos(self.angle) * power * signed_thrust
            self.vy += math.sin(self.angle) * power * signed_thrust

        # 3. Drag
        self.vx *= self.drag
        self.vy *= self.drag

        # 4. Gravity
        self.vy += self.gravity

        # 4b. Top-speed clamp
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
    def __init__(self, config: SimulationConfig, evasion=True, rng=None, n_obstacles=0):
        # Store the config first
        self.cfg = config
        self.evasion = evasion
        self.rng = rng or random.Random()
        self.n_obstacles = n_obstacles
        self.obstacles = []

        # Pull target overrides directly from the config object
        self.target_speed = self.cfg.target_stats['target_speed']
        self.target_turn_speed = self.cfg.target_stats['turn_speed']
        self.target_thrust_power = self.cfg.target_stats['thrust_power']

        self.reset()

    def _spawn_point(self):
        """A point at least SPAWN_WALL_MARGIN from every wall."""
        m = SPAWN_WALL_MARGIN
        return (self.rng.uniform(m, WIDTH - m), self.rng.uniform(m, HEIGHT - m))

    def _spawn_near(self, px, py):
        """A point [SPAWN_MIN_SEP, SPAWN_MAX_SEP] from (px,py)."""
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
        # Pass the config object into the Drones
        px, py = self._spawn_point()
        self.player = Drone(px, py, config=self.cfg)

        tx, ty = self._spawn_near(px, py)
        self.target = Drone(tx, ty, config=self.cfg)
        
        # Override Target specifics
        self.target.top_speed = self.target_speed
        self.target.turn_speed = self.target_turn_speed
        self.target.thrust_power = self.target_thrust_power

        self.target_wp_x, self.target_wp_y = self._spawn_point()

        if self.n_obstacles > 0:
            import objects
            n = self.rng.randint(1, self.n_obstacles)
            avoid = [(self.player.x, self.player.y, DRONE_RADIUS),
                     (self.target.x, self.target.y, DRONE_RADIUS)]
            self.obstacles = objects.spawn_obstacles(self.rng, n, avoid)
        else:
            self.obstacles = []

    def step(self, actions, auto_reset=True):
        """Advance one frame."""
        if isinstance(actions, dict):
            self.player.apply_physics(
                thrust=actions.get('thrust', False),
                reverse=actions.get('reverse', False),
                left=actions.get('left', False),
                right=actions.get('right', False),
            )
        else:
            self.player.apply_physics_continuous(actions[0], actions[1])

        # Target AI
        dist_to_player = math.hypot(self.player.x - self.target.x,
                                    self.player.y - self.target.y)

        if self.evasion and dist_to_player < EVASION_TRIGGER_RADIUS:
            escape_angle = math.atan2(self.target.y - self.player.y,
                                      self.target.x - self.player.x)
            self.target_wp_x, self.target_wp_y = _ray_to_boundary(
                self.target.x, self.target.y, escape_angle, DRONE_RADIUS)

        dist_to_wp = math.hypot(self.target_wp_x - self.target.x,
                                self.target_wp_y - self.target.y)
        if dist_to_wp < WAYPOINT_REACHED_DIST:
            self.target_wp_x, self.target_wp_y = self._spawn_point()

        wp_angle = math.atan2(self.target_wp_y - self.target.y,
                              self.target_wp_x - self.target.x)
        self.target.x += math.cos(wp_angle) * self.target_speed
        self.target.y += math.sin(wp_angle) * self.target_speed

        # State data
        distance = dist_to_player
        target_angle = math.atan2(self.target.y - self.player.y,
                                  self.target.x - self.player.x)
        relative_angle = (target_angle - self.player.angle + math.pi) % (2 * math.pi) - math.pi

        # Collision & Escape
        is_collision = distance < (DRONE_RADIUS * 2)

        target_escaped = self.evasion and (
            self.target.x <= WALL_OFFSET or
            self.target.x >= WIDTH - WALL_OFFSET or
            self.target.y <= WALL_OFFSET or
            self.target.y >= HEIGHT - WALL_OFFSET)

        player_crashed = (
            self.player.x <= DRONE_RADIUS or
            self.player.x >= WIDTH - DRONE_RADIUS or
            self.player.y <= DRONE_RADIUS or
            self.player.y >= HEIGHT - DRONE_RADIUS)

        if (is_collision or target_escaped or player_crashed) and auto_reset:
            self.reset()

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