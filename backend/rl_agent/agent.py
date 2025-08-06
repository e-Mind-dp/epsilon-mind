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
            return action
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state)
        action = q_values.argmax().item()
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

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self):
        self.update()

