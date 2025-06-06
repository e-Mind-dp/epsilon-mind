# # rl_agent/config.py

# # Epsilon action space (i.e., what RL agent can pick from)
# EPSILON_ACTIONS = [round(x * 0.1, 1) for x in range(1, 21)]  # [0.1, 0.2, ..., 2.0]

# # Neural network hyperparameters
# STATE_SIZE = 5              # [query_type, sensitivity, remaining_budget, similarity, user_type]
# ACTION_SIZE = len(EPSILON_ACTIONS)
# HIDDEN_SIZE = 64

# # Training hyperparameters
# LEARNING_RATE = 1e-3
# GAMMA = 0.99
# BATCH_SIZE = 64
# REPLAY_BUFFER_SIZE = 10000
# MIN_REPLAY_SIZE = 1000
# TARGET_UPDATE_FREQ = 100
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 5000  # steps
# MAX_EPISODES = 1000
# MAX_STEPS_PER_EPISODE = 50


# config.py
EPSILON_ACTIONS = [round(x * 0.1, 1) for x in range(1, 21)]  # [0.1, 0.2, ..., 2.0]

STATE_DIM = 6               # 6 input features
ACTION_SPACE = len(EPSILON_ACTIONS)
ACTION_DIM = len(EPSILON_ACTIONS)

EPISODES = 1000
MAX_STEPS = 10              # Per episode
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10
