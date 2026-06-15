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

WIDTH, HEIGHT = 800, 600
DRONE_RADIUS = 30

# ---------------------------------------------------------------------------
# Target evasion AI -- tuning knobs (see comments in DroneSimulator.step)
# ---------------------------------------------------------------------------
EVASION_TRIGGER_RADIUS = 200   # player within this distance (px) spooks the target
ESCAPE_LOOKAHEAD = 300         # how far ahead (px) the target projects its flee waypoint
WAYPOINT_REACHED_DIST = 20     # distance (px) at which a waypoint counts as "reached"
TARGET_SPEED = 3.0             # target travel speed (px/frame), constant -- no inertia


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

    def apply_physics(self, thrust=False, reverse=False, left=False, right=False):
        # 1. Orientation (pitch)
        if left:
            self.angle -= self.turn_speed
        if right:
            self.angle += self.turn_speed

        # 2. Forward thrust along heading
        if thrust:
            self.vx += math.cos(self.angle) * self.thrust_power
            self.vy += math.sin(self.angle) * self.thrust_power

        # 3. Reverse thrust (push back along heading)
        if reverse:
            self.vx -= math.cos(self.angle) * self.thrust_power
            self.vy -= math.sin(self.angle) * self.thrust_power

        # 4. Drag on the drone's own motion (applied before gravity)
        self.vx *= self.drag
        self.vy *= self.drag

        # 5. Gravity = constant downward acceleration
        self.vy += self.gravity

        # 6. Integrate position
        self.x += self.vx
        self.y += self.vy

        # 7. Boundary collisions
        if self.x < DRONE_RADIUS:
            self.x, self.vx = DRONE_RADIUS, 0
        if self.x > WIDTH - DRONE_RADIUS:
            self.x, self.vx = WIDTH - DRONE_RADIUS, 0
        if self.y < DRONE_RADIUS:
            self.y, self.vy = DRONE_RADIUS, 0
        if self.y > HEIGHT - DRONE_RADIUS:
            self.y, self.vy = HEIGHT - DRONE_RADIUS, 0


class DroneSimulator:
    def __init__(self, evasion=True, target_speed=TARGET_SPEED, rng=None):
        # `evasion` / `target_speed` let the RL side dial difficulty (curriculum).
        # `rng` lets training seed reproducibly without touching global random.
        self.evasion = evasion
        self.target_speed = target_speed
        self.rng = rng or random.Random()
        self.reset()

    def _rand_point(self, lo, hi_w, hi_h):
        return (self.rng.randint(lo, hi_w), self.rng.randint(lo, hi_h))

    def reset(self):
        self.player = Drone(WIDTH / 4, HEIGHT / 2)

        tx, ty = self._rand_point(100, WIDTH - 100, HEIGHT - 100)
        self.target = Drone(tx, ty)

        wx, wy = self._rand_point(100, WIDTH - 100, HEIGHT - 100)
        self.target_wp_x, self.target_wp_y = wx, wy

    def step(self, actions, auto_reset=True):
        """Advance one frame.

        actions:    dict with bool keys thrust/reverse/left/right
        auto_reset: True for the live game (snap to a new round on intercept).
                    The RL env passes False so it can end the episode itself.
        """
        # 1. Player physics
        self.player.apply_physics(
            thrust=actions.get('thrust', False),
            reverse=actions.get('reverse', False),
            left=actions.get('left', False),
            right=actions.get('right', False),
        )

        # =================================================================
        # 2. TARGET EVASION AI  (A) FLEE  (B) WANDER  (C) MOVE
        # =================================================================
        dist_to_player = math.hypot(self.player.x - self.target.x,
                                    self.player.y - self.target.y)

        # (A) FLEE: if threatened, plot an escape waypoint directly away.
        if self.evasion and dist_to_player < EVASION_TRIGGER_RADIUS:
            escape_angle = math.atan2(self.target.y - self.player.y,
                                      self.target.x - self.player.x)
            self.target_wp_x = self.target.x + math.cos(escape_angle) * ESCAPE_LOOKAHEAD
            self.target_wp_y = self.target.y + math.sin(escape_angle) * ESCAPE_LOOKAHEAD
            self.target_wp_x = max(DRONE_RADIUS, min(WIDTH - DRONE_RADIUS, self.target_wp_x))
            self.target_wp_y = max(DRONE_RADIUS, min(HEIGHT - DRONE_RADIUS, self.target_wp_y))

        # (B) WANDER: reached the waypoint? pick a fresh random one.
        dist_to_wp = math.hypot(self.target_wp_x - self.target.x,
                                self.target_wp_y - self.target.y)
        if dist_to_wp < WAYPOINT_REACHED_DIST:
            self.target_wp_x, self.target_wp_y = self._rand_point(
                100, WIDTH - 100, HEIGHT - 100)

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
        if is_collision and auto_reset:
            self.reset()

        return {
            "drone": {"x": self.player.x, "y": self.player.y, "angle": self.player.angle},
            "target": {"x": self.target.x, "y": self.target.y},
            "state": {"distance": distance, "relative_angle": relative_angle},
            "collision": is_collision,
        }
