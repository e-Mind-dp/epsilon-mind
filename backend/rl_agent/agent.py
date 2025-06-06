# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# from q_network import QNetwork
# from replay_buffer import ReplayBuffer 

# class DDQNAgent:
#     def __init__(
#         self,
#         state_size,
#         action_size,
#         epsilon_values,
#         learning_rate=1e-3,
#         gamma=0.95,
#         epsilon_start=1.0,
#         epsilon_min=0.05,
#         epsilon_decay=0.995,
#         batch_size=64,
#         memory_size=10000,
#         target_update_freq=10,
#         device="cpu"
#     ):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.epsilon_values = epsilon_values
#         self.gamma = gamma  # discount factor
#         self.epsilon = epsilon_start
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.batch_size = batch_size
#         self.device = device
#         self.target_update_freq = target_update_freq

#         # Experience replay memory
#         # self.memory = deque(maxlen=memory_size)
#         self.memory = ReplayBuffer(capacity=memory_size)

#         # Q-Networks
#         self.policy_net = QNetwork(state_size, action_size).to(device)
#         self.target_net = QNetwork(state_size, action_size).to(device)
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

#         # Sync target initially
#         self.update_target_network()

#         self.step_counter = 0

#     def update_target_network(self):
#         """Sync weights from policy network to target network."""
#         self.target_net.load_state_dict(self.policy_net.state_dict())

#     def remember(self, state, action, reward, next_state, done):
#         """Store an experience in replay buffer."""
#         # self.memory.append((state, action, reward, next_state, done))
#         self.memory.push(state, action, reward, next_state, done)

#     def act(self, state):
#         """Choose action using epsilon-greedy policy."""
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)  # explore
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             q_values = self.policy_net(state_tensor)
#         return torch.argmax(q_values).item()  # exploit

#     def replay(self):
#         """Train the policy network from a batch of experiences."""
#         if len(self.memory) < self.batch_size:
#             return  # Not enough experiences yet

#         # Sample mini-batch
#         # minibatch = random.sample(self.memory, self.batch_size)
#         # states, actions, rewards, next_states, dones = zip(*minibatch)

#         states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

#         states = torch.FloatTensor(states).to(self.device)
#         next_states = torch.FloatTensor(next_states).to(self.device)
#         actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
#         rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
#         dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

#         # Q-values for current states
#         current_q = self.policy_net(states).gather(1, actions)

#         # --- DDQN: Get best action from policy_net, then evaluate it using target_net ---
#         next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
#         next_q = self.target_net(next_states).gather(1, next_actions)

#         # Compute target Q-value
#         target_q = rewards + (1 - dones) * self.gamma * next_q

#         # Loss: MSE between current and target Q
#         loss = nn.MSELoss()(current_q, target_q.detach())

#         # Backpropagation
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # Decay exploration rate
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#         # Periodically update target network
#         self.step_counter += 1
#         if self.step_counter % self.target_update_freq == 0:
#             self.update_target_network()








# agent.py

import torch
import numpy as np
import torch.optim as optim
from rl_agent.q_network import DQN
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.config import *

class DDQNAgent:
    def __init__(self, input_dim, output_dim, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, tau=1e-2, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-networks
        self.q_net = DQN(input_dim, output_dim)
        self.target_net = DQN(input_dim, output_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.action_counts = [0] * output_dim  # Track action frequencies


    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(self.output_dim)
            # print(f"Random action: {action}")
            return action
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state)
        # print(f"Q-values: {q_values.detach().numpy()}")
        action = q_values.argmax().item()
        # print(f"Greedy action: {action}")
        # return q_values.argmax().item()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_value = self.q_net(states)
        q_values = q_value.gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q_values * (~dones)

        # q_values = self.q_net(states).gather(1, actions)
        # next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
        # next_q_values = self.target_net(next_states).gather(1, next_actions)
        # target_q = rewards + self.gamma * next_q_values * (~dones)

        # --- Entropy regularization to promote exploration ---
        entropy = -torch.sum(torch.softmax(q_values, dim=1) * torch.log_softmax(q_values, dim=1), dim=1).mean()
        loss = torch.nn.functional.mse_loss(q_values, target_q) - 0.01 * entropy  # exploration bonus


        # loss = torch.nn.functional.mse_loss(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Epsilon decay ---
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self):
        self.update()

