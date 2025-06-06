# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from q_network import QNetwork
# from replay_buffer import ReplayBuffer
# from env_sim import DpEnvironment
# import matplotlib.pyplot as plt


# # Hyperparameters
# STATE_SIZE = 5
# ACTION_SIZE = 5
# BATCH_SIZE = 16
# GAMMA = 0.99
# LEARNING_RATE = 0.001
# BUFFER_SIZE = 1000
# EPISODES = 50

# # Epsilon-greedy params (for exploration vs exploitation)
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 0.995

# # Define action space (modifiers)
# ACTION_MODIFIERS = [0.5, 0.75, 1.0, 1.25, 1.5]

# # Initialize everything
# env = DpEnvironment()
# q_net = QNetwork(STATE_SIZE, ACTION_SIZE)
# target_net = QNetwork(STATE_SIZE, ACTION_SIZE)
# target_net.load_state_dict(q_net.state_dict())  # sync weights
# optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
# replay_buffer = ReplayBuffer(BUFFER_SIZE)

# epsilon = EPSILON_START

# reward_history = []

# for episode in range(EPISODES):
#     state = env.reset()
#     total_reward = 0

#     done = False
#     while not done:
#         # ε-greedy action selection
#         if np.random.rand() < epsilon:
#             action_idx = np.random.randint(ACTION_SIZE)
#         else:
#             with torch.no_grad():
#                 state_tensor = torch.FloatTensor(state)
#                 q_values = q_net(state_tensor)
#                 action_idx = torch.argmax(q_values).item()

#         modifier = ACTION_MODIFIERS[action_idx]

#         # Take action in env
#         next_state, reward, done = env.step(modifier)

#         # Store in replay buffer
#         replay_buffer.push(state, action_idx, reward, next_state, done)
#         state = next_state
#         total_reward += reward

#         # Train only if buffer has enough data
#         if len(replay_buffer) >= BATCH_SIZE:
#             s_batch, a_batch, r_batch, ns_batch, d_batch = replay_buffer.sample(BATCH_SIZE)

#             s_batch = torch.FloatTensor(s_batch)
#             a_batch = torch.LongTensor(a_batch)
#             r_batch = torch.FloatTensor(r_batch)
#             ns_batch = torch.FloatTensor(ns_batch)
#             d_batch = torch.FloatTensor(d_batch)

#             # Q(s, a)
#             q_values = q_net(s_batch)
#             q_selected = q_values.gather(1, a_batch.unsqueeze(1)).squeeze()

#             # Q_target(s’, a’) ← using target network
#             next_q_values = target_net(ns_batch)
#             next_q_max = next_q_values.max(1)[0]
#             q_target = r_batch + GAMMA * next_q_max * (1 - d_batch)

#             # Loss & backprop
#             loss = nn.MSELoss()(q_selected, q_target.detach())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     # Decay ε
#     epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

#     print(f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

#     reward_history.append(total_reward)


# plt.plot(reward_history)
# plt.title("Training Reward over Episodes")
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.grid(True)
# plt.savefig("training_rewards.png")  # Save as image
# plt.show()

# # Save trained model weights
# torch.save(q_net.state_dict(), "trained_q_net.pth")
# print("✅ Q-network model saved to trained_q_net.pth")






# main.py

# from agent import DDQNAgent
# from dp_environment import EpsilonEnv
# from utils import encode_state, get_epsilon_from_action
# from config import EPISODES, MAX_STEPS

# env = EpsilonEnv()
# agent = DDQNAgent()

# for episode in range(EPISODES):
#     state = env.reset()
#     for step in range(MAX_STEPS):
#         action_idx = agent.act(state)
#         epsilon = get_epsilon_from_action(action_idx)
#         next_state, reward, done = env.step(state, epsilon)
#         agent.memory.add((state, action_idx, reward, next_state, done))
#         agent.update()
#         state = next_state

#         if done:
#             break

#     if episode % 10 == 0:
#         agent.update_target()
#         print(f"Episode {episode} complete — epsilon: {agent.epsilon:.3f}")









# main.py

import numpy as np
import random
import torch
from collections import Counter

from agent import DDQNAgent
from dp_environment import DPEnvironment
from utils import *
from config import *

 

def sample_random_query(budget):
    sensitivities = ["low", "medium", "high", "extreme"]
    roles = ["doctor", "researcher", "public"]
    query_types = ["individual", "aggregate", "filtering", "temporal", "comparative", "descriptive", "unknown"]
    
    return {
        "sensitivity": random.choice(sensitivities),
        "role": random.choice(roles),
        "query_type": random.choice(query_types),
        "similarity": round(random.uniform(0, 1), 2),
        "confidence": round(random.uniform(0, 1), 2),  # Optional: enforce minimum confidence
        "budget": budget  # Start with current budget
    }


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    env = DPEnvironment()
    agent = DDQNAgent(input_dim=6, output_dim=20)  
    episodes = 500000
    episode_rewards = []
    action_counter = Counter()
    set_seed(42)


    # Epsilon-greedy setup
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05

    for ep in range(episodes):
        # === Randomly sample a query ===
        state = env.reset()
        query = sample_random_query(env.budget)
        state = encode_state(
            query["sensitivity"], query["role"], query["query_type"],
            query["similarity"], query["budget"], query["confidence"]
        )

        done = False
        total_reward = 0

        while not done:
            action_index = agent.select_action(state, epsilon)
            # action_index = agent.select_action(state)
            # epsilon = 0.1 + action_index * 0.2
            # epsilon_val = get_epsilon_from_action(action_index)
            epsilon_val = get_epsilon_from_action(action_index)
            action_counter[action_index] += 1
            next_state, reward, done, info = env.step(state, epsilon_val)

            # print(f"[Ep {ep}] Mode: {info['mode']}, Query: {query}, Epsilon: {epsilon:.2f}, Reward: {reward:.4f}")

            experience = state, action_index, reward, next_state, done

            agent.replay_buffer.add(experience)
            agent.train()

            state = next_state
            total_reward += reward
        
        # Decay exploration
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if ep % 10000 == 0 and ep < 500000:
            print(f"[Ep {ep}] Reward: {reward:.4f} | Mode: {info['mode']} | Bonus: {info['bonus']:.3f} | Utility: {info['utility']:.3f} | Privacy Cost: {info['privacy_cost']:.3f}")


        # if ep % 500 == 0:
        #     print(f"[Ep {ep}] Total Reward: {total_reward:.3f} | Epsilon: {epsilon:.3f}")


        # print(f"Episode {ep} completed with Total Reward: {total_reward:.4f}\n")

    # Save rewards for plotting
    np.save("episode_rewards.npy", np.array(episode_rewards))

    # === Save the trained Q-network weights ===
    torch.save(agent.q_net.state_dict(), "ddqn_q_network.pth")
    print("✅ Q-network weights saved to ddqn_q_network.pth")


    # # === Load model for manual test (optional, ensures you're using saved weights) ===
    # agent.q_net.load_state_dict(torch.load("ddqn_q_network.pth"))
    # agent.q_net.eval()

    # # === Manual Test Case ===
    # print("\n🔍 Manual Test Case")
    # test_query = {
    #     "sensitivity": "low",
    #     "role": "public",
    #     "query_type": "aggregate",
    #     "similarity": 0.10,
    #     "confidence": 0.95,
    #     "budget": 10.0
    # }


    # test_state = encode_state(
    #     test_query["sensitivity"],
    #     test_query["role"],
    #     test_query["query_type"],
    #     test_query["similarity"],
    #     test_query["budget"],
    #     test_query["confidence"]
    # )

    # # Use deterministic behavior for manual test (set epsilon to 0 for pure exploitation)
    # test_epsilon = 0.0

    # with torch.no_grad():
    #     q_values = agent.q_net(torch.FloatTensor(test_state).unsqueeze(0))
    #     q_values = q_values.squeeze().tolist()
    #     for i, q in enumerate(q_values):
    #         eps = get_epsilon_from_action(i)
    #         print(f"Action {i:2d} → Epsilon: {eps:.2f} → Q-value: {q:.4f}")

    # action_index = agent.select_action(test_state, test_epsilon)
    # epsilon_val = get_epsilon_from_action(action_index)
    # # action_index = agent.select_action(test_state)
    # # epsilon = get_epsilon_from_action(action_index)

    # print(f"Test Query: {test_query}")
    # print(f"→ Chosen epsilon value: {epsilon_val:.2f} (Action Index: {action_index})")


if __name__ == "__main__":
    main()



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









