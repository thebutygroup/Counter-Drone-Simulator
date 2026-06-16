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
from drone_core import SimulationConfig, cfg

# Observation layout (see drone_env._obs):
# [dist_norm, rel_angle, heading, vx, vy, tvx, tvy, px, py]
_OBS_LOW = np.array([0.0, -math.pi, -math.pi, -60, -60, -60, -60, 0.0, 0.0], dtype=np.float32)
_OBS_HIGH = np.array([1.5,  math.pi,  math.pi,  60,  60,  60,  60, 1.0, 1.0], dtype=np.float32)


def _wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class DroneGymEnv(gym.Env):
    metadata = {"render_modes": []}

    # 1. Swap `target_speed` for `config: SimulationConfig = None`
    def __init__(self, evasion=True, closing_scale=0.10, config: SimulationConfig = None, 
                 seed=None, n_obstacles=0, perception="none", perception_kwargs=None,
                 continuous=False):
        super().__init__()
        self.evasion = evasion
        self.closing_scale = closing_scale
        
        # 2. Store the config (fallback to master cfg if none provided)
        self.cfg = config or cfg
        
        self._seed = seed
        self.n_obstacles = n_obstacles
        self.perception = perception
        self.perception_kwargs = perception_kwargs or {}
        self.continuous = continuous
        
        # Continuous: 2-D Box [thrust, turn] in [-1,1] (PPO Gaussian policy).
        # Discrete: the legacy 7-action set (tabular / DQN / discrete-PPO baseline).
        if continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(N_ACTIONS)
            
        self._build()
        
        # Observation space = base 9-D + whatever the chosen perception appends.
        plo, phi = self.env.perception.bounds()
        low = np.concatenate([_OBS_LOW, plo]) if plo.size else _OBS_LOW
        high = np.concatenate([_OBS_HIGH, phi]) if phi.size else _OBS_HIGH
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def _build(self):
            """Helper to construct/reconstruct the underlying headless environment."""
            # 1. NO MORE `if self.target_speed is not None:` block! 
            #    Delete any of that old override logic. The config handles it all.
            
            # 2. Just instantiate the base environment with the config.
            self.env = DroneInterceptEnv(
                config=self.cfg,               # Pass the config object!
                evasion=self.evasion,
                seed=self._seed,
                n_obstacles=self.n_obstacles,
                perception=self.perception,
                perception_kwargs=self.perception_kwargs,
                continuous=self.continuous
            )

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
        act = action if self.continuous else int(action)
        obs, reward, done, info = self.env.step(act)
        terminated = bool(info["success"] or info["crashed"] or info.get("escaped"))
        truncated = bool(done and not terminated)              # timeout
        return self._fix(obs), float(reward), terminated, truncated, info
