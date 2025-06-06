# import numpy as np
# import random

# class DpEnvironment:
#     def __init__(self, epsilon_values):
#         self.epsilon_values = epsilon_values  # Action space: list of allowed epsilons (e.g. [0.1, 0.2, ..., 2.0])
#         self.state = None
#         self.query_count = 0

#     def reset(self, state):
#         """Initialize the environment with the first query's state."""
#         self.state = np.array(state, dtype=np.float32)
#         self.query_count = 0
#         return self.state

#     def step(self, action_index):
#         """Take action (choose epsilon), return next state, reward, and done."""
#         chosen_epsilon = self.epsilon_values[action_index]

#         # Unpack the current state
#         sensitivity, remaining_budget, query_type, similarity, role_cap = self.state

#         # ----------------------
#         # 🎯 REWARD FUNCTION
#         # ----------------------
#         # GOAL: Reward should reflect a smart tradeoff between:
#         # - Privacy (low epsilon)
#         # - Utility (high epsilon)
#         # - Budget conservation (not wasting epsilon)
#         # - Sensitivity handling (sensitive query → lower epsilon)
#         # - Familiarity (similar queries will be assigned lower epsilon)

#         # Penalty if chosen epsilon too high for high sensitivity
#         if sensitivity == 2:  # high
#             privacy_penalty = max(0, chosen_epsilon - 0.3) * 2
#         elif sensitivity == 1:  # medium
#             privacy_penalty = max(0, chosen_epsilon - 0.8)
#         else:  # low
#             privacy_penalty = 0  # no privacy risk



#         # Utility reward: reward slightly for higher epsilon (but less for sensitive queries)
#         if sensitivity == 0:
#             utility_reward = chosen_epsilon * 1.0
#         elif sensitivity == 1:
#             utility_reward = chosen_epsilon * 0.6
#         else:
#             utility_reward = chosen_epsilon * 0.3



#         # Budget penalty: avoid using up too much budget if it's low
#         budget_penalty = 0
#         if remaining_budget < 0.2:
#             budget_penalty = chosen_epsilon * 1.5



#         # Penalize high epsilon when similarity is high (redundant query)
#         if similarity >= 0.85:
#             similarity_penalty = similarity * chosen_epsilon * 1.5
#         else:
#             similarity_penalty = 0

#         # Final reward
#         reward = utility_reward - similarity_penalty - privacy_penalty - budget_penalty


#         # Update state (simulate budget change)
#         updated_remaining_budget = max(0.0, remaining_budget - chosen_epsilon / role_cap)

#         # Construct new state (only budget changes here)
#         new_state = np.array([
#             sensitivity,
#             updated_remaining_budget,
#             query_type,
#             similarity,
#             role_cap
#         ], dtype=np.float32)

#         self.state = new_state
#         self.query_count += 1

#         # Training episode ends after fixed number of queries
#         done = self.query_count >= 100 
        
#         return new_state, reward, done







# import random

# class EpsilonEnv:
#     def __init__(self):
#         pass

#     def reset(self):
#         # Dummy starting state
#         return [random.randint(0, 2),  # sensitivity
#                 random.randint(0, 2),  # role
#                 random.randint(0, 2),  # query_type
#                 random.uniform(0, 1),  # semantic_similarity
#                 random.uniform(0.5, 1.0),  # remaining_budget
#                 random.uniform(0.6, 1.0)]  # sensitivity_confidence

#     def step(self, state, epsilon):
#         # Dummy logic for now
#         sensitivity, _, _, similarity, budget, _ = state
#         reward = (1.0 - similarity) * (1.0 - epsilon / 2.0)  # High reward if less similarity + lower epsilon
#         done = budget < 0.2
#         next_state = self.reset()
#         return next_state, reward, done







# dp_environment.py

import numpy as np

class DPEnvironment:
    def __init__(self):
        self.query_history = []  # Store semantic embeddings
        self.budget = 1.0  # Full budget at start

    def reset(self):
        self.budget = 1.0
        self.query_history = []
        return self._get_state()

    def _get_state(self):
        # Placeholder: you will pass this from outside
        return None



    def _select_mode(self, state):
        sensitivity, user_role, query_type, similarity, remaining_budget, confidence = state

        if similarity > 0.85:
            return "anti_abuse"
        
        if sensitivity >= 2:
            if user_role == 0:  # doctor
                return "balanced"
            else:
                return "privacy_critical"
            
        if sensitivity <= 1:
            return "utility_focused"
        
        if remaining_budget < 2.0:
            return "privacy_critical"
        
        return "balanced"



    def _get_weights(self, mode):
        weight_map = {
            # "balanced": (0.6, 0.3, 0.1),
            # "privacy_critical": (0.4, 0.5, 0.1),
            # "utility_focused": (0.7, 0.2, 0.1),
            # "anti_abuse": (0.5, 0.3, 0.2),
            "balanced": (0.8, 0.15, 0.05),
            "utility_focused": (0.9, 0.05, 0.05),
            "privacy_critical": (0.5, 0.4, 0.1),  # slightly reduce privacy weight
            "anti_abuse": (0.5, 0.3, 0.2)         # this one’s fine
        }
        return weight_map[mode]

    def step(self, state, epsilon):
        sensitivity, user_role, query_type, similarity, remaining_budget, confidence = state

        # --- Mode Selection ---
        mode = self._select_mode(state)
        w1, w2, w3 = self._get_weights(mode)

        utility = confidence * np.exp(-1 / epsilon) # proxy for answer quality
        privacy_cost = epsilon  # higher epsilon → more leakage
        redundancy_penalty = similarity  # higher similarity = worse

        # --- Rule-based reward shaping ---
        bonus = 0.0

        # Rule 1: Encourage higher epsilon if low sensitivity
        if sensitivity == 0 and epsilon >= 0.8:
            bonus += 2.0 * epsilon

        # Rule 2: Penalize if high similarity (redundant query) and high epsilon
        if similarity > 0.8 and epsilon > 0.5:
            bonus -= 0.2 * epsilon

        # Rule 3: If high sensitivity but role is 'doctor' (0), allow moderate epsilon
        if sensitivity >= 2 and user_role == 0 and 0.2 <= epsilon <= 0.6:
            bonus += 0.1 * epsilon

        # Rule 4: If high sensitivity and public (2), penalize even moderate epsilon
        if sensitivity >= 2 and user_role == 2 and epsilon > 0.3:
            bonus -= 2.0 * epsilon

        if sensitivity == 0:
            if epsilon >= 0.8:
                if similarity >= 0.85:
                    bonus -= 0.2 * epsilon
                else:
                    bonus += 2.0 * epsilon  # Encourage bold action when safe


        # --- Final Reward ---
        raw_reward = (w1 * utility) - (w2 * privacy_cost) - (w3 * redundancy_penalty) + bonus

        # --- Update budget ---

        self.budget -= epsilon
        self.budget = max(0.0, self.budget)
        new_budget = self.budget

        # --- Update query history ---
        self.query_history.append(similarity)  

        done = self.budget <= 0.01

        # Build new state with updated budget
        next_state = [
            sensitivity,
            user_role,
            query_type,
            similarity,
            new_budget,
            confidence
        ]

        reward = np.tanh(raw_reward)

        # return self._get_state(), reward, done, {"mode": mode, "weights": (w1, w2, w3)}
        return next_state, reward, done, {
        "mode": mode,
        "weights": (w1, w2, w3),
        "bonus": bonus,
        "utility": utility,
        "privacy_cost": privacy_cost,
        "redundancy": redundancy_penalty,
        "raw_reward": raw_reward,
        "final_reward": reward
    }