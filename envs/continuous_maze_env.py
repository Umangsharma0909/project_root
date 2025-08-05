import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ContinuousMazeEnv(gym.Env):
    """Continuous control maze with varying friction/terrain."""
    def __init__(self, size=10, terrain_scale=0.5, seed=None):
        super().__init__()
        self.size = size
        self.terrain = np.random.uniform(0.1, terrain_scale, (size, size))
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(0, size-1, shape=(2,), dtype=np.float32)
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self, **kwargs):
        self.agent_pos = np.array([0.0, 0.0])
        return self.agent_pos, {}

    def step(self, action):
        # action determines velocity vector
        friction = self.terrain[int(self.agent_pos[0]), int(self.agent_pos[1])]
        self.agent_pos = np.clip(self.agent_pos + action * (1 - friction), 0, self.size-1)
        done = np.linalg.norm(self.agent_pos - np.array([self.size-1, self.size-1])) < 0.5
        reward = 1.0 if done else -np.linalg.norm(action)
        return self.agent_pos, reward, done, False, {}
