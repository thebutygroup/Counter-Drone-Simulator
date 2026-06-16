"""
perception.py  (core)
=====================
Pluggable obstacle perception for the interceptor's observation. The env owns ONE
Perception instance and appends its output to the fixed 9-D ego/target vector.

Design contract:
  * OBSERVATION is pluggable (raycast ring vs nearest-K slots vs none).
  * REWARD is NOT -- obstacle collisions/penalties are computed from geometry in
    drone_env, independent of perception, so swapping modes keeps runs comparable.

Each Perception exposes:
  obs_dim()          -> int, number of appended observation scalars
  bounds()           -> (low[obs_dim], high[obs_dim]) float32 arrays for the Box
  reset(rng)         -> per-episode setup (e.g. ray jitter phase)
  observe(sim)       -> float32[obs_dim]

All perceptions consume the shared collision primitive from objects.py: every
obstacle is a list of circles (cx, cy, r). Walls are handled by the raycast mode
(unified ranging) and ignored by slots (the base obs already carries px/py).

Frames of reference: outputs are EGOCENTRIC (rotated into the drone's heading),
matching the base obs's heading-relative target bearing.
"""

import math
import numpy as np

from drone_core import WIDTH, HEIGHT, DRONE_RADIUS

# Defaults (overridable via perception_kwargs)
DEFAULT_N_RAYS = 16
DEFAULT_RAY_RANGE = 500.0     # px; beyond this a ray reads "clear" (1.0)
DEFAULT_SLOTS = 4
DEFAULT_SLOT_RANGE = 600.0    # px; normaliser for slot offsets/radii


class Perception:
    name = "none"
    def obs_dim(self): return 0
    def bounds(self):
        z = np.zeros(0, dtype=np.float32)
        return z, z
    def reset(self, rng): pass
    def observe(self, sim): return np.zeros(0, dtype=np.float32)


class NoPerception(Perception):
    """Obstacles invisible to the agent (default). Obs stays the base 9-D."""
    name = "none"


class RaycastPerception(Perception):
    """A ring of N egocentric distance sensors ('2D lidar'). Each ray returns the
    normalised distance to the nearest surface -- WALL or obstacle -- along its
    heading-relative bearing. Count-agnostic; unifies walls + obstacles; a 1-D
    depth image that bridges toward the FPV/vision stage."""
    name = "rays"

    def __init__(self, n_rays=DEFAULT_N_RAYS, max_range=DEFAULT_RAY_RANGE, jitter=True):
        self.n = int(n_rays)
        self.max_range = float(max_range)
        self.jitter = jitter
        self._phase = 0.0

    def obs_dim(self): return self.n
    def bounds(self):
        return (np.zeros(self.n, dtype=np.float32), np.ones(self.n, dtype=np.float32))

    def reset(self, rng):
        # small per-episode angular offset so a thin pole never sits permanently
        # invisible between two fixed rays
        self._phase = rng.uniform(0, 2 * math.pi / self.n) if self.jitter else 0.0

    def observe(self, sim):
        p = sim.player
        out = np.empty(self.n, dtype=np.float32)
        circles = [c for o in sim.obstacles for c in o["collision"]]
        for i in range(self.n):
            ang = p.angle + self._phase + (2 * math.pi * i / self.n)
            dx, dy = math.cos(ang), math.sin(ang)
            d = self._ray_to_wall(p.x, p.y, dx, dy)
            for (cx, cy, r) in circles:
                t = self._ray_to_circle(p.x, p.y, dx, dy, cx, cy, r)
                if t < d:
                    d = t
            out[i] = min(d, self.max_range) / self.max_range
        return out

    def _ray_to_wall(self, ox, oy, dx, dy):
        # drone is inside [0,W]x[0,H]; find distance to the rectangle boundary
        cand = []
        if dx > 1e-9:   cand.append((WIDTH - ox) / dx)
        elif dx < -1e-9: cand.append((0 - ox) / dx)
        if dy > 1e-9:   cand.append((HEIGHT - oy) / dy)
        elif dy < -1e-9: cand.append((0 - oy) / dy)
        return min([c for c in cand if c > 0], default=self.max_range)

    def _ray_to_circle(self, ox, oy, dx, dy, cx, cy, r):
        fx, fy = ox - cx, oy - cy
        b = dx * fx + dy * fy
        c = fx * fx + fy * fy - r * r
        disc = b * b - c
        if disc < 0:
            return self.max_range
        sq = math.sqrt(disc)
        t = -b - sq
        if t >= 0:
            return t
        t2 = -b + sq
        return 0.0 if t2 >= 0 else self.max_range   # origin inside the circle


class SlotPerception(Perception):
    """Nearest-K obstacles as egocentric (present, dx, dy, r), padded with zeros.
    Simple, MLP-friendly, exact geometry; capped at K entities."""
    name = "slots"

    def __init__(self, k=DEFAULT_SLOTS, max_range=DEFAULT_SLOT_RANGE):
        self.k = int(k)
        self.max_range = float(max_range)

    def obs_dim(self): return self.k * 4
    def bounds(self):
        lo = np.tile([0.0, -1.0, -1.0, 0.0], self.k).astype(np.float32)
        hi = np.tile([1.0, 1.0, 1.0, 1.0], self.k).astype(np.float32)
        return lo, hi

    def observe(self, sim):
        p = sim.player
        ca, sa = math.cos(-p.angle), math.sin(-p.angle)   # rotate world -> heading frame
        items = []
        for o in sim.obstacles:
            # represent the obstacle by its nearest collision circle
            cx, cy, r = min(o["collision"], key=lambda c: math.hypot(c[0]-p.x, c[1]-p.y))
            wx, wy = cx - p.x, cy - p.y
            ex = wx * ca - wy * sa      # egocentric x (forward = +x along heading)
            ey = wx * sa + wy * ca
            items.append((math.hypot(wx, wy), ex, ey, r))
        items.sort(key=lambda it: it[0])

        out = np.zeros(self.k * 4, dtype=np.float32)
        for s in range(min(self.k, len(items))):
            _, ex, ey, r = items[s]
            base = s * 4
            out[base + 0] = 1.0
            out[base + 1] = float(np.clip(ex / self.max_range, -1, 1))
            out[base + 2] = float(np.clip(ey / self.max_range, -1, 1))
            out[base + 3] = float(np.clip(r / self.max_range, 0, 1))
        return out


def make_perception(name, **kwargs):
    name = (name or "none").lower()
    if name in ("none", "off"):
        return NoPerception()
    if name in ("rays", "raycast", "lidar"):
        return RaycastPerception(**kwargs)
    if name in ("slots", "entity", "slot"):
        return SlotPerception(**kwargs)
    raise ValueError(f"unknown perception '{name}' (use none|rays|slots)")
