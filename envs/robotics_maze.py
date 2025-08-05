import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DegradingMazeEnv(gym.Env):
    """Maze navigation with actuator degradation over time."""
    def __init__(self, size=10, degrade_rate=0.01, seed=None):
        super().__init__()
        self.size = size
        self.degrade_rate = degrade_rate
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(0, size-1, shape=(2,), dtype=np.int32)
        self.degradation = 1.0
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self, **kwargs):
        self.agent_pos = np.array([0, 0])
        self.degradation = 1.0
        return self.agent_pos, {}

    def step(self, action):
        # degrade actuator reliability
        self.degradation = max(0.0, self.degradation - self.degrade_rate)
        effective_action = action if np.random.rand() < self.degradation else self.action_space.sample()
        # move agent
        moves = {0: [0,1], 1: [0,-1], 2: [1,0], 3: [-1,0]}
        self.agent_pos = np.clip(self.agent_pos + moves[effective_action], 0, self.size-1)
        done = np.array_equal(self.agent_pos, [self.size-1, self.size-1])
        reward = 1.0 if done else -0.01
        return self.agent_pos, reward, done, False, {}
