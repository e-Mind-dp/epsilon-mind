# import torch
# import torch.nn as nn
# # from config import STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE


# class QNetwork(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=64):
#         super(QNetwork, self).__init__()

#         # Simple 2-layer MLP (Multilayer Perceptron)
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, action_size)

#         # self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
#         # self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
#         # self.out = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)


#     def forward(self, state):
#         """
#         Forward pass: input is the state (vector), output is Q-values for all epsilon actions
#         """
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         q_values = self.out(x)
#         return q_values



import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)
