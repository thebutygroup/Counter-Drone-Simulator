"""
test_continuous.py  (core)
==========================
Pure-numpy unit tests for the lever-1 continuous action space. NO torch / SB3 /
gymnasium required -- exercises drone_core (physics) and drone_env (Gym-style
env) directly, so the control + reward math is verified before PPO instability
can confound it.

Run from the core/ folder:
    python test_continuous.py
or with pytest:
    pytest test_continuous.py -q
"""

import math
import numpy as np

import drone_core
from drone_core import Drone, TOP_SPEED
import drone_env
from drone_env import DroneInterceptEnv

TP = 0.45      # Drone.thrust_power
DRAG = 0.99    # Drone.drag
GRAV = 0.06    # Drone.gravity
TS = 0.10      # Drone.turn_speed
TOL = 1e-6


def _fresh(angle=0.0):
    """A drone parked away from the walls so boundary clamps don't interfere."""
    d = Drone(drone_core.WIDTH / 2, drone_core.HEIGHT / 2, angle=angle)
    return d


# ---------------------------------------------------------------------------
# 1. Continuous thrust: sign = direction, magnitude scales linearly.
# ---------------------------------------------------------------------------
def test_thrust_sign_and_magnitude():
    # Full forward at heading 0 -> +x velocity = thrust_power, then drag, +gravity on y.
    d = _fresh(0.0)
    d.apply_physics_continuous(1.0, 0.0)
    assert abs(d.vx - TP * DRAG) < TOL, d.vx
    assert abs(d.vy - GRAV) < TOL, d.vy

    # Full reverse -> mirror of forward in x.
    d = _fresh(0.0)
    d.apply_physics_continuous(-1.0, 0.0)
    assert abs(d.vx - (-TP * DRAG)) < TOL, d.vx

    # Half thrust -> exactly half the forward impulse (magnitude modulation,
    # impossible in the discrete set).
    d = _fresh(0.0)
    d.apply_physics_continuous(0.5, 0.0)
    assert abs(d.vx - 0.5 * TP * DRAG) < TOL, d.vx
    print("ok  test_thrust_sign_and_magnitude")


# ---------------------------------------------------------------------------
# 2. Continuous turn: sign = direction (+right / -left), magnitude scales rate.
# ---------------------------------------------------------------------------
def test_turn_sign_and_magnitude():
    d = _fresh(0.0); d.apply_physics_continuous(0.0, 1.0)
    assert abs(d.angle - TS) < TOL, d.angle
    d = _fresh(0.0); d.apply_physics_continuous(0.0, -1.0)
    assert abs(d.angle - (-TS)) < TOL, d.angle
    d = _fresh(0.0); d.apply_physics_continuous(0.0, 0.5)
    assert abs(d.angle - 0.5 * TS) < TOL, d.angle
    print("ok  test_turn_sign_and_magnitude")


# ---------------------------------------------------------------------------
# 3. Discrete is BIT-FOR-BIT preserved, and equals the matching continuous call.
#    (Guarantees the discrete baseline didn't move under the refactor.)
# ---------------------------------------------------------------------------
def test_discrete_equivalence():
    # coast
    a, b = _fresh(0.3), _fresh(0.3)
    a.apply_physics()                      # discrete coast
    b.apply_physics_continuous(0.0, 0.0)   # continuous coast
    assert (a.x, a.y, a.vx, a.vy, a.angle) == (b.x, b.y, b.vx, b.vy, b.angle)

    # thrust + right  ==  continuous(+1, +1)
    a, b = _fresh(0.3), _fresh(0.3)
    a.apply_physics(thrust=True, right=True)
    b.apply_physics_continuous(1.0, 1.0)
    assert (a.x, a.y, a.vx, a.vy, a.angle) == (b.x, b.y, b.vx, b.vy, b.angle)

    # reverse + left  ==  continuous(-1, -1)
    a, b = _fresh(-0.7), _fresh(-0.7)
    a.apply_physics(reverse=True, left=True)
    b.apply_physics_continuous(-1.0, -1.0)
    assert (a.x, a.y, a.vx, a.vy, a.angle) == (b.x, b.y, b.vx, b.vy, b.angle)
    print("ok  test_discrete_equivalence")


# ---------------------------------------------------------------------------
# 4. THE lever-1 capability: reverse-thrust AND rotate in the SAME frame.
#    Discrete cannot (no reverse+turn action, and the 7-set never combines them);
#    continuous can. Verify the heading rotated AND velocity points opposite the
#    new heading -- i.e. "moving backwards relative to where you point", the exact
#    precondition for a rear-first approach at speed.
# ---------------------------------------------------------------------------
def test_simultaneous_reverse_and_turn():
    d = _fresh(0.0)
    d.apply_physics_continuous(-1.0, 1.0)          # reverse + turn right, one frame
    assert abs(d.angle - TS) < TOL, d.angle        # heading DID rotate this frame
    hx, hy = math.cos(d.angle), math.sin(d.angle)  # unit heading
    vdoth = d.vx * hx + d.vy * hy                  # velocity projected on heading
    assert vdoth < 0.0, vdoth                      # moving backwards w.r.t. heading
    print("ok  test_simultaneous_reverse_and_turn")


# ---------------------------------------------------------------------------
# 5. Top-speed magnitude clamp holds every frame under sustained full thrust.
# ---------------------------------------------------------------------------
def test_top_speed_clamp():
    # Invariant: magnitude never exceeds the cap under sustained full thrust,
    # for any heading (wall clamps only ever REDUCE speed, so this is safe).
    for ang in (0.0, -math.pi / 2, 0.9):
        d = _fresh(ang)
        for _ in range(400):
            d.apply_physics_continuous(1.0, 0.3)
            assert math.hypot(d.vx, d.vy) <= TOP_SPEED * (1 + 1e-9)
    # Activation: an over-cap velocity is pulled exactly onto the cap in one frame
    # (coast action, mid-arena so no wall interferes).
    d = _fresh(0.0); d.vx, d.vy = 100.0, 0.0
    d.apply_physics_continuous(0.0, 0.0)
    assert abs(math.hypot(d.vx, d.vy) - TOP_SPEED) < TOL, math.hypot(d.vx, d.vy)
    print("ok  test_top_speed_clamp")


# ---------------------------------------------------------------------------
# 6. Out-of-range actions are clamped to the box (unsquashed Gaussian safety).
# ---------------------------------------------------------------------------
def test_action_clamp():
    a, b = _fresh(0.4), _fresh(0.4)
    a.apply_physics_continuous(5.0, -9.0)          # way outside [-1,1]
    b.apply_physics_continuous(1.0, -1.0)          # the clamped equivalent
    assert (a.x, a.y, a.vx, a.vy, a.angle) == (b.x, b.y, b.vx, b.vy, b.angle)
    print("ok  test_action_clamp")


# ---------------------------------------------------------------------------
# 7. Continuous env: obs shape/finiteness, Box-action step, and the discrete
#    path still runs unchanged through the same env class.
# ---------------------------------------------------------------------------
def test_env_continuous_and_discrete():
    env = DroneInterceptEnv(evasion=True, seed=0, continuous=True)
    obs = env.reset()
    assert obs.shape == (DroneInterceptEnv.BASE_OBS_DIM,), obs.shape
    assert np.all(np.isfinite(obs))

    for act in ([0.5, -0.3], [10.0, -10.0], [-1.0, 1.0]):   # incl. out-of-range
        obs, r, done, info = env.step(np.asarray(act, dtype=np.float32))
        assert obs.shape == (DroneInterceptEnv.BASE_OBS_DIM,)
        assert np.all(np.isfinite(obs)) and math.isfinite(r)
        assert set(("success", "crashed", "escaped", "impact_speed", "backside")) <= set(info)
        if done:
            break

    # discrete path unaffected
    denv = DroneInterceptEnv(evasion=True, seed=0, continuous=False)
    obs = denv.reset()
    obs, r, done, info = denv.step(1)              # action index 1 = thrust
    assert obs.shape == (DroneInterceptEnv.BASE_OBS_DIM,) and math.isfinite(r)
    print("ok  test_env_continuous_and_discrete")


# ---------------------------------------------------------------------------
# 8. Reward ORDERING on the live knobs (mirrors drone_env.step's collision block).
#    The control objective: fast+rear-first is the prize; speed and orientation
#    each independently help; slow prop-first is worst. We assert those robust
#    relations and PRINT all four so the (slow-rear vs fast-prop) tension created
#    by the additive orient bonus is visible -- that's the thing to watch when
#    you re-tune magnitudes for continuous control.
# ---------------------------------------------------------------------------
def _intercept_bonus(impact_speed, backside, steps):
    de = drone_env
    speed_gate = float(np.clip(
        (impact_speed - de.MIN_IMPACT_SPEED) /
        (de.FULL_IMPACT_SPEED - de.MIN_IMPACT_SPEED), 0.0, 1.0))
    time_frac = max(0.0, 1.0 - steps / de.PAR_STEPS)
    return (de.INTERCEPT_BONUS
            + de.IMPACT_SPEED_BONUS * impact_speed * speed_gate
            + de.IMPACT_ORIENT_BONUS * backside
            + de.BACKSIDE_JACKPOT * speed_gate * backside
            + de.TIME_BONUS * time_frac)


def test_reward_ordering():
    steps = 60
    fast_rear = _intercept_bonus(9.0, 1.0, steps)
    fast_prop = _intercept_bonus(9.0, 0.0, steps)
    slow_rear = _intercept_bonus(3.0, 1.0, steps)
    slow_prop = _intercept_bonus(3.0, 0.0, steps)
    print(f"    bonuses  fast_rear={fast_rear:.1f}  fast_prop={fast_prop:.1f}  "
          f"slow_rear={slow_rear:.1f}  slow_prop={slow_prop:.1f}")

    vals = {"fast_rear": fast_rear, "fast_prop": fast_prop,
            "slow_rear": slow_rear, "slow_prop": slow_prop}
    assert max(vals, key=vals.get) == "fast_rear"   # the prize is the unique max
    assert min(vals, key=vals.get) == "slow_prop"   # slow prop-first is worst
    assert fast_rear > fast_prop                    # orientation pays at speed
    assert fast_rear > slow_rear                    # speed pays at good orientation
    assert fast_prop > slow_prop                    # speed pays at bad orientation
    print("ok  test_reward_ordering")


if __name__ == "__main__":
    test_thrust_sign_and_magnitude()
    test_turn_sign_and_magnitude()
    test_discrete_equivalence()
    test_simultaneous_reverse_and_turn()
    test_top_speed_clamp()
    test_action_clamp()
    test_env_continuous_and_discrete()
    test_reward_ordering()
    print("\nALL CONTINUOUS-ACTION TESTS PASSED")
