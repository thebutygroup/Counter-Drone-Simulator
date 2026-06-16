"""
objects.py  (core)
==================
The 2D obstacle library. Each object type is defined ONCE in real-world METRES
(canonical SPECS below) and converted to pixels via PIXELS_PER_METER. The same
metric numbers carry straight to phase 2 (PyBullet, metre-native), so the 2D and
future 3D libraries stay dimensionally identical.

Every object provides two things:
  * COLLISION  -- a list of circles (cheap, uniform). Used for hit tests in 2D
                  and trivially portable to ray-circle queries (raycast obs) or
                  to PyBullet primitive collision shapes in 3D.
  * RENDER     -- rough drawing primitives (line / circle / rect) for the viewer.
                  "Very rough shapes", as intended -- not detailed meshes.

Coordinates in a spec are (x_right, y_up) in METRES relative to the object's base,
which sits on the ground (bottom wall). The builder converts to screen pixels:
    sx = base_x + x_right * M * scale
    sy = HEIGHT  - y_up    * M * scale      (screen y is DOWN, ground at HEIGHT)
"""

import math

from drone_core import WIDTH, HEIGHT, DRONE_RADIUS, PIXELS_PER_METER

M = PIXELS_PER_METER

OBJECT_TYPES = ["pole", "street_lamp", "tree", "person", "dog"]

# Each spec: nominal height (m, for reference), a colour, COLLISION circles
# (x_right, y_up, r) in metres, and RENDER primitives in metres.
# Render primitive kinds: ("line", x1,y1,x2,y2,width) | ("circle", cx,cy,r) |
#                         ("rect", cx,cy,halfw,halfh)
SPECS = {
    "pole": {
        "color": "#8a8f98",
        "collision": [(0, 0.6, 0.3), (0, 1.8, 0.3), (0, 3.0, 0.3), (0, 3.8, 0.3)],
        "render": [("line", 0, 0, 0, 4.0, 0.35)],
    },
    "street_lamp": {
        "color": "#c9b458",
        "collision": [(0, 0.6, 0.3), (0, 1.8, 0.3), (0, 3.0, 0.3),
                      (0, 4.2, 0.3), (0.7, 4.7, 0.5)],
        "render": [("line", 0, 0, 0, 5.0, 0.35),       # mast
                   ("line", 0, 5.0, 0.8, 4.8, 0.25),   # arm
                   ("circle", 0.85, 4.7, 0.45)],       # lamp head
    },
    "tree": {
        "color": "#5a8f5a",
        "collision": [(0, 0.6, 0.4), (0, 1.5, 0.4),    # trunk
                      (0, 2.9, 1.4), (-0.7, 2.4, 0.9), (0.7, 2.4, 0.9)],  # canopy
        "render": [("line", 0, 0, 0, 2.2, 0.5),        # trunk
                   ("circle", 0, 2.9, 1.4),            # canopy blobs
                   ("circle", -0.8, 2.3, 0.9),
                   ("circle", 0.8, 2.3, 0.9)],
    },
    "person": {
        "color": "#d98a8a",
        "collision": [(0, 0.5, 0.35), (0, 1.1, 0.35), (0, 1.62, 0.22)],
        "render": [("circle", 0, 1.62, 0.2),           # head
                   ("line", 0, 1.4, 0, 0.7, 0.4),      # torso
                   ("line", 0, 0.7, -0.25, 0, 0.25),   # legs
                   ("line", 0, 0.7, 0.25, 0, 0.25)],
    },
    "dog": {
        "color": "#b08d57",
        "collision": [(-0.25, 0.35, 0.3), (0.2, 0.35, 0.3), (0.5, 0.5, 0.2)],
        "render": [("circle", -0.05, 0.38, 0.32),      # body
                   ("circle", 0.5, 0.5, 0.2),          # head
                   ("line", -0.3, 0.2, -0.3, 0, 0.12), # legs
                   ("line", 0.2, 0.2, 0.2, 0, 0.12)],
    },
}


def make_obstacle(otype, base_x, scale=1.0):
    """Build one obstacle instance at ground x=base_x, scaled. Returns a dict
    with absolute-pixel collision circles + render primitives + colour."""
    spec = SPECS[otype]
    s = scale * M

    def px(xr, yup):
        return (base_x + xr * s, HEIGHT - yup * s)

    collision = [(*px(xr, yup), r * s) for (xr, yup, r) in spec["collision"]]

    render = []
    for prim in spec["render"]:
        kind = prim[0]
        if kind == "line":
            x1, y1 = px(prim[1], prim[2])
            x2, y2 = px(prim[3], prim[4])
            render.append({"kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                           "w": prim[5] * s})
        elif kind == "circle":
            cx, cy = px(prim[1], prim[2])
            render.append({"kind": "circle", "cx": cx, "cy": cy, "r": prim[3] * s})
        elif kind == "rect":
            cx, cy = px(prim[1], prim[2])
            render.append({"kind": "rect", "cx": cx, "cy": cy,
                           "hw": prim[3] * s, "hh": prim[4] * s})

    top = min(c[1] - c[2] for c in collision)   # highest point (smallest screen y)
    return {"type": otype, "color": spec["color"], "base_x": base_x,
            "scale": scale, "collision": collision, "render": render, "top_y": top}


def obstacle_hit(obs, x, y, radius):
    """True if a circle of `radius` at (x,y) overlaps any of the obstacle's
    collision circles."""
    for (cx, cy, r) in obs["collision"]:
        if math.hypot(x - cx, y - cy) <= r + radius:
            return True
    return False


def obstacle_clearance(obs, x, y):
    """Signed-ish nearest gap (px) from point (x,y) to the obstacle surface;
    negative inside. Used later for a dense proximity penalty (mirrors walls)."""
    best = float("inf")
    for (cx, cy, r) in obs["collision"]:
        best = min(best, math.hypot(x - cx, y - cy) - r)
    return best


def spawn_obstacles(rng, n, avoid, margin=80, pad=40, tries=60):
    """Place `n` obstacles at random ground x-positions, scaled 0.7-1.3x.
    `avoid` is a list of (x, y, r) the obstacles must not overlap (drone spawns).
    Obstacles also avoid overlapping each other. Returns a list (may be < n if
    the arena is too crowded to place them all)."""
    placed = []
    for _ in range(n):
        otype = rng.choice(OBJECT_TYPES)
        scale = rng.uniform(0.7, 1.3)
        ok = False
        for _ in range(tries):
            bx = rng.uniform(margin, WIDTH - margin)
            cand = make_obstacle(otype, bx, scale)
            clash = any(obstacle_hit(cand, ax, ay, ar + pad) for (ax, ay, ar) in avoid)
            if not clash:
                for other in placed:
                    if any(obstacle_hit(cand, cx, cy, cr + pad)
                           for (cx, cy, cr) in other["collision"]):
                        clash = True
                        break
            if not clash:
                placed.append(cand)
                ok = True
                break
        # if we couldn't place this one without clashing, just skip it
    return placed
