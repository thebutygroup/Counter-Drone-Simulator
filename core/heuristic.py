"""
heuristic.py  (core)
====================
A model-based pursuit controller used as the EXPERT for behaviour cloning.

It is the exact velocity-matching policy from the web demo, ported to Python:
plan a target VELOCITY pointed at the lead point, then steer/thrust (including
reverse-thrust) to reach it. Because drag is light (~0.99) the drone coasts a
long way, so the controller must actively brake -- it eases the desired speed
down near the evader and near walls, and reverse-thrusts when mis-aligned.

Returns a CONTINUOUS action [thrust, turn], each in [-1, 1] -- i.e. exactly the
DroneGymEnv Box action space (continuous=True). Use it to generate expert
trajectories for BC (see stage1_2d_profile/pretrain_bc.py), which drops a PPO
policy straight into the "pursue directly" basin and skips the loiter trap.
"""

import math

from drone_core import WIDTH, HEIGHT, DRONE_RADIUS


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# tuned coefficients (see web build; ~97% intercept on identical physics)
LEAD_MAX = 22.0
CLOSE_K = 0.5          # ease desired speed down within ~2x dist of the evader
CLOSE_BIAS = 8.0
WALL_K = 0.12         # ease desired speed down near walls (must brake to stop)
WALL_BIAS = 4.0
OBST_RANGE = 260.0     # obstacle repulsion reach (px)
OBST_GAIN = 1.8
OBST_SLOW = 90.0       # cut desired speed when an obstacle is this close


def expert_action(player, target, tvx, tvy, obstacles,
                  top_speed, thrust_power, turn_speed):
    """Compute the expert's [thrust, turn] for the current frame.

    player : object with .x .y .vx .vy .angle   (sim.player)
    target : object with .x .y                   (sim.target)
    tvx,tvy: evader velocity this frame (finite difference, px/frame)
    obstacles: list of dicts with "collision" = [(cx,cy,r), ...]  (may be empty)
    top_speed, thrust_power, turn_speed: the live config values
    """
    p, t = player, target
    dx, dy = t.x - p.x, t.y - p.y
    dist = math.hypot(dx, dy)
    speed = math.hypot(p.vx, p.vy)

    # lead the aim point by the evader's velocity, more when it's far
    lead = _clamp(dist / max(speed, 4.0), 0.0, LEAD_MAX)
    aimx, aimy = t.x + tvx * lead, t.y + tvy * lead

    wall_dist = min(p.x, WIDTH - p.x, p.y, HEIGHT - p.y)
    desired = min(top_speed, dist * CLOSE_K + CLOSE_BIAS, wall_dist * WALL_K + WALL_BIAS)

    ang = math.atan2(aimy - p.y, aimx - p.x)
    tvX, tvY = math.cos(ang) * desired, math.sin(ang) * desired

    # obstacle repulsion: bend the desired velocity away from nearby collision
    # circles, and ease off speed when one is close, so it routes around trees.
    for o in obstacles:
        for (cx, cy, cr) in o["collision"]:
            gap = math.hypot(p.x - cx, p.y - cy) - cr - DRONE_RADIUS
            if gap < OBST_RANGE:
                k = 1.0 - max(gap, 0.0) / OBST_RANGE
                a2 = math.atan2(p.y - cy, p.x - cx)
                tvX += math.cos(a2) * desired * OBST_GAIN * k
                tvY += math.sin(a2) * desired * OBST_GAIN * k
                if gap < OBST_SLOW:
                    desired *= 0.6

    # velocity error -> needed acceleration; steer toward it, thrust by magnitude
    dvx, dvy = tvX - p.vx, tvY - p.vy
    acc_dir = math.atan2(dvy, dvx)
    acc_mag = math.hypot(dvx, dvy)

    da = _wrap(acc_dir - p.angle)
    turn = _clamp(da / max(turn_speed, 1e-3), -1.0, 1.0)
    align = math.cos(da)
    if align > 0.25:
        thrust = _clamp(acc_mag / thrust_power, 0.0, 1.0)
    elif align < -0.4:
        thrust = -_clamp(acc_mag / thrust_power, 0.0, 1.0)   # reverse-thrust to brake/redirect
    else:
        thrust = 0.0
    return thrust, turn
