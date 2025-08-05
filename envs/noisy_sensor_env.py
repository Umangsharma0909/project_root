import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NoisySensorEnv(gym.Env):
    """Environment with sensor noise and drift to test robustness."""
    def __init__(self, base_env, noise_std=0.1, drift_rate=0.001, seed=None):
        super().__init__()
        self.base_env = base_env
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        self.drift = 0.0
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed=None, options=None):
        """Resets the base environment and applies noise/drift."""
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.drift = 0.0
        noisy = self._add_noise(obs)
        return noisy, info
        obs, info = self.base_env.reset()
        self.drift = 0.0
        noisy = self._add_noise(obs)
        return noisy, info

    def step(self, action):
        obs, reward, done, truncated, info = self.base_env.step(action)
        self.drift += self.drift_rate
        noisy_obs = self._add_noise(obs)
        return noisy_obs, reward, done, truncated, info

    def _add_noise(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise + self.drift
