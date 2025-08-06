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
    roles = ["Authorised", "Unauthorised"]
    query_types = ["individual", "aggregate", "filtering", "temporal", "comparative", "descriptive", "unknown"]
    
    return {
        "sensitivity": random.choice(sensitivities),
        "role": random.choice(roles),
        "query_type": random.choice(query_types),
        "similarity": round(random.uniform(0, 1), 2),
        "confidence": round(random.uniform(0, 1), 2),  
        "budget": budget  
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
            epsilon_val = get_epsilon_from_action(action_index)
            action_counter[action_index] += 1
            next_state, reward, done, info = env.step(state, epsilon_val)

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

    # Save rewards for plotting
    np.save("episode_rewards.npy", np.array(episode_rewards))

    # === Save the trained Q-network weights ===
    torch.save(agent.q_net.state_dict(), "ddqn_q_network.pth")
    print("✅ Q-network weights saved to ddqn_q_network.pth")


if __name__ == "__main__":
    main()

