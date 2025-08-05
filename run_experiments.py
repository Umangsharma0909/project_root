from envs.robotics_maze import DegradingMazeEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
import numpy as np

if __name__ == "__main__":
    env = DegradingMazeEnv(size=10, degrade_rate=0.01)
    dqn = DQNAgent(env)
    dqn.train(200_000)
    dqn_rewards = dqn.evaluate(env)

    ppo = PPOAgent(env)
    ppo.train(200_000)
    ppo_rewards = ppo.evaluate(env)

    print("DQN mean reward:", np.mean(dqn_rewards))
    print("PPO mean reward:", np.mean(ppo_rewards))