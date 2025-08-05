import pandas as pd
import matplotlib.pyplot as plt

# Simulated results table
df = pd.DataFrame({
    'Environment': [
        'DegradingMaze', 'DegradingMaze', 'DynamicObstacle', 'DynamicObstacle',
        'NoisySensor(DQN)', 'NoisySensor(PPO)', 'ContinuousMaze', 'ContinuousMaze'],
    'Agent': ['DQN','PPO']*4,
    'MeanReward': [0.72,0.85,0.60,0.78,0.65,0.82,0.55,0.75]
})

fig, ax = plt.subplots()
for agent in df['Agent'].unique():
    subset = df[df['Agent']==agent]
    ax.plot(subset['Environment'], subset['MeanReward'], marker='o', label=agent)
ax.set_ylabel('Mean Reward')
ax.set_title('Virtual Agent Performance Comparison')
ax.legend()
plt.show()
