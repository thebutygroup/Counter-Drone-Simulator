"""
gym_env.py  (core)
==================
Gymnasium adapter around DroneInterceptEnv so Stable-Baselines3 (DQN/PPO) can
train on the *same* 2D physics the tabular agent used.

Key adaptations vs the raw env:
  * Modern Gymnasium API: reset() -> (obs, info); step() -> (obs, reward,
    terminated, truncated, info).
  * terminated vs truncated done correctly: an intercept or a wall-crash is a
    real terminal state (terminated=True); a timeout is truncation
    (truncated=True). DQN bootstraps on truncation but not on termination, so
    this distinction matters for correctness.
  * Angles wrapped to [-pi, pi] so the network sees bounded, periodic inputs
    (the raw drone heading is unbounded as it spins).
  * closing_scale set per-env so a curriculum (survive -> catch -> evade) can be
    run by swapping envs between learn() calls.
"""

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import drone_env
from drone_env import DroneInterceptEnv, N_ACTIONS

# Observation layout (see drone_env._obs):
# [dist_norm, rel_angle, heading, vx, vy, tvx, tvy, px, py]
_OBS_LOW = np.array([0.0, -math.pi, -math.pi, -60, -60, -60, -60, 0.0, 0.0], dtype=np.float32)
_OBS_HIGH = np.array([1.5,  math.pi,  math.pi,  60,  60,  60,  60, 1.0, 1.0], dtype=np.float32)


def _wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class DroneGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, evasion=True, closing_scale=0.10, target_speed=None, seed=None, backside_gate=False):
        super().__init__()
        self.backside_gate = backside_gate
        self.evasion = evasion
        self.closing_scale = closing_scale
        self.target_speed = target_speed
        self._seed = seed
        self.observation_space = spaces.Box(_OBS_LOW, _OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._build()
        drone_env.USE_BACKSIDE_GATE = self.backside_gate

    def _build(self):
        kwargs = {}
        if self.target_speed is not None:
            kwargs["target_speed"] = self.target_speed
        self.env = DroneInterceptEnv(evasion=self.evasion, seed=self._seed, **kwargs)

    def _fix(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        obs[2] = _wrap(obs[2])           # heading -> [-pi, pi]
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Only rebuild (reseed) when an explicit seed is given; otherwise let the
        # env's RNG advance so successive episodes differ.
        if seed is not None:
            self._seed = seed
            self._build()
        drone_env.CLOSING_SCALE = self.closing_scale
        obs = self.env.reset()
        return self._fix(obs), {}

    def step(self, action):
        drone_env.CLOSING_SCALE = self.closing_scale          # set reward regime
        obs, reward, done, info = self.env.step(int(action))
        terminated = bool(info["success"] or info["crashed"])  # real terminal
        truncated = bool(done and not terminated)              # timeout
        return self._fix(obs), float(reward), terminated, truncated, info
