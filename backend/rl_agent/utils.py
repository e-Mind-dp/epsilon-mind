# utils.py
import json
import os
from rl_agent.config import EPSILON_ACTIONS

def encode_state(sensitivity, user_role, query_type, similarity, budget, confidence):
    # print(f"DEBUG: sensitivity={sensitivity} ({type(sensitivity)})")
    # print(f"DEBUG: user_role={user_role} ({type(user_role)})")
    # print(f"DEBUG: query_type={query_type} ({type(query_type)})")


    role_map = {"doctor": 0, "researcher": 1, "public": 2}
    query_map = {
        "individual": 0, "aggregate": 1, "filtering": 2,
        "temporal": 3, "comparative": 4, "descriptive": 5, "unknown": 6
    }
    sensitivity_map = {"low": 0, "medium": 1, "high": 2, "extreme": 3}

     # If user_role is int, convert to string by reverse mapping
    if isinstance(user_role, int):
        reverse_role_map = {v: k for k, v in role_map.items()}
        user_role = reverse_role_map.get(user_role, "public")  # default to "public" or raise error

    # Standardize inputs
    sensitivity = sensitivity.lower().strip()
    user_role = user_role.lower().strip()
    query_type = query_type.lower().strip()

    return [
        float(sensitivity_map[sensitivity]),
        float(role_map[user_role]),
        float(query_map[query_type]),
        float(similarity),
        float(budget),
        float(confidence)
    ]



def get_epsilon_from_action(action_idx):
    return EPSILON_ACTIONS[action_idx]

# Load optimized epsilon list once at module level
with open(os.path.join("best_epsilon_strategy.json"), "r") as f:
    DE_PSO_EPSILONS = json.load(f)

def optimised_epsilon(opt_actions):
    return DE_PSO_EPSILONS[opt_actions]
