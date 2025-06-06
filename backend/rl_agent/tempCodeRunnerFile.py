
import numpy as np
import matplotlib.pyplot as plt


rewards = np.load("episode_rewards.npy")

def moving_average(x, w=100):
    return np.convolve(x, np.ones(w), 'valid') / w

plt.figure(figsize=(10, 6))
plt.plot(moving_average(rewards), label="Smoothed Reward", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.title("DDQN Agent Reward Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("RL Agent Training Reward Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()









