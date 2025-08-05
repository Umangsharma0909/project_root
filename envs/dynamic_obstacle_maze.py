import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DynamicObstacleMazeEnv(gym.Env):
    """Maze with moving obstacles requiring re-planning."""
    def __init__(self, size=10, n_obstacles=5, obstacle_speed=1, seed=None):
        super().__init__()
        self.size = size
        self.n_obstacles = n_obstacles
        self.obstacle_speed = obstacle_speed
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(0, size-1, shape=(2,), dtype=np.int32),
            'obstacles': spaces.Box(0, size-1, shape=(n_obstacles,2), dtype=np.int32)
        })
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self, seed=None, options=None):
        """Resets the environment and initializes obstacles."""
        self.agent_pos = np.array([0,0])
        self.obstacles = np.random.randint(0, self.size, size=(self.n_obstacles,2))
        return {'agent': self.agent_pos, 'obstacles': self.obstacles}, {}
        self.agent_pos = np.array([0,0])
        self.obstacles = np.random.randint(0, self.size, size=(self.n_obstacles,2))
        return {'agent': self.agent_pos, 'obstacles': self.obstacles}, {}

    def step(self, action):
        # move agent
        moves = {0:[0,1],1:[0,-1],2:[1,0],3:[-1,0]}
        self.agent_pos = np.clip(self.agent_pos + moves[action], 0, self.size-1)
        # move obstacles randomly
        self.obstacles = np.clip(
            self.obstacles + np.random.randint(-self.obstacle_speed, self.obstacle_speed+1, self.obstacles.shape),
            0, self.size-1)
        # check collisions
        if any((self.agent_pos == obs).all() for obs in self.obstacles):
            return {'agent':self.agent_pos,'obstacles':self.obstacles}, -1.0, True, False, {}
        done = (self.agent_pos == [self.size-1,self.size-1]).all()
        reward = 1.0 if done else -0.01
        return {'agent':self.agent_pos,'obstacles':self.obstacles}, reward, done, False, {}
