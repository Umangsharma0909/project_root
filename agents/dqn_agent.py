from stable_baselines3 import DQN

class DQNAgent:
    def __init__(self, env):
        self.model = DQN("MlpPolicy", env, verbose=1)
    def train(self, timesteps=100_000):
        self.model.learn(total_timesteps=timesteps)
    def evaluate(self, env, episodes=10):
        rewards = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False; ep_r = 0
            while not done:
                action, _ = self.model.predict(obs)
                obs, r, done, _, _ = env.step(action)
                ep_r += r
            rewards.append(ep_r)
        return rewards