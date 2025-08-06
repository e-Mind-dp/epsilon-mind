import numpy as np

class DPEnvironment:
    def __init__(self):
        self.query_history = []  
        self.budget = 1.0  

    def reset(self):
        self.budget = 1.0
        self.query_history = []
        return self._get_state()

    def _get_state(self):
        return None



    def _select_mode(self, state):
        sensitivity, user_role, query_type, similarity, remaining_budget, confidence = state

        if similarity > 0.85:
            return "anti_abuse"
        
        if sensitivity >= 2:
            if user_role == 0:  
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
            "balanced": (0.8, 0.15, 0.05),
            "utility_focused": (0.9, 0.05, 0.05),
            "privacy_critical": (0.5, 0.4, 0.1),  
            "anti_abuse": (0.5, 0.3, 0.2)         
        }
        return weight_map[mode]

    def step(self, state, epsilon):
        sensitivity, user_role, query_type, similarity, remaining_budget, confidence = state

        # --- Mode Selection ---
        mode = self._select_mode(state)
        w1, w2, w3 = self._get_weights(mode)

        utility = confidence * np.exp(-1 / epsilon) 
        privacy_cost = epsilon  
        redundancy_penalty = similarity 

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