import os
import logging
from datetime import datetime
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from envs.robotics_maze import DegradingMazeEnv
from envs.dynamic_obstacle_maze import DynamicObstacleMazeEnv
from envs.noisy_sensor_env import NoisySensorEnv
from envs.continuous_maze_env import ContinuousMazeEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent

# 1) Setup logging directory and file
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# 2) Callback for logging episode rewards
def wrap_callback():
    class RewardLogger(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
        def _on_step(self) -> bool:
            if 'episode' in self.locals.get('infos', [{}])[0]:
                ep_info = self.locals['infos'][0]['episode']
                logging.info(f"Finished episode: reward={ep_info['r']}, length={ep_info['l']}")
            return True
    return RewardLogger()

# 3) Runner function
def run_and_log(env, AgentClass, name, timesteps=200_000):
    logging.info(f"Starting {name} on {env.__class__.__name__}")
    agent = AgentClass(env)
    callback = wrap_callback()
    agent.model.learn(total_timesteps=timesteps, callback=callback)
    rewards = agent.evaluate(env)
    mean_r, std_r = np.mean(rewards), np.std(rewards)
    logging.info(f"Completed {name}: mean_reward={mean_r:.3f}, std_reward={std_r:.3f}")
    return mean_r, std_r

if __name__ == "__main__":
    experiments = [
        (DegradingMazeEnv(), DQNAgent, "DQN"),
        (DegradingMazeEnv(), PPOAgent, "PPO"),
        (DynamicObstacleMazeEnv(), DQNAgent, "DQN"),
        (DynamicObstacleMazeEnv(), PPOAgent, "PPO"),
        (NoisySensorEnv(DegradingMazeEnv()), DQNAgent, "DQN"),
        (NoisySensorEnv(DegradingMazeEnv()), PPOAgent, "PPO"),
        (ContinuousMazeEnv(), DQNAgent, "DQN"),
        (ContinuousMazeEnv(), PPOAgent, "PPO"),
    ]
    results = {}
    for env, AgentClass, name in experiments:
        key = f"{env.__class__.__name__}_{name}"
        results[key] = run_and_log(env, AgentClass, name)

    # Save summary CSV
    import pandas as pd
    df = pd.DataFrame([
        {"experiment": k, "mean_reward": v[0], "std_reward": v[1]}
        for k, v in results.items()
    ])
    summary_path = os.path.join(log_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    logging.info(f"Summary saved to {summary_path}")
