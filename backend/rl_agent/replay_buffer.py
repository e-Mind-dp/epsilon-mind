# # replay_buffer.py

# import random
# from collections import deque
# import numpy as np

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)  # stores transitions

#     def push(self, state, action, reward, next_state, done):
#         # Save a single experience
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         # Randomly sample a batch of experiences
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         return (
#             np.array(states),
#             np.array(actions),
#             np.array(rewards),
#             np.array(next_states),
#             np.array(dones)
#         )

#     def __len__(self):
#         return len(self.buffer)






import random
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
