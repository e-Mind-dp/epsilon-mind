import numpy as np

def simulate_user_response(epsilon_policy, user_model="adversarial"):
    """
    Simulates how a strategic user would react to the given epsilon policy.
    They will submit queries to maximize total privacy leakage.
    """
    if user_model == "honest":
        # Honest user issues random queries
        query_indices = np.random.choice(len(epsilon_policy), size=5, replace=False)
    else:
        # Adversarial user selects top-k epsilons to maximize leakage
        query_indices = np.argsort(epsilon_policy)[-5:]

    cumulative_leakage = np.sum(epsilon_policy[query_indices])
    return cumulative_leakage


def compute_fitness(epsilon_policy):
    """
    Convert leakage to a fitness score for DE–PSO (maximize negative leakage).
    """
    leakage = simulate_user_response(epsilon_policy, user_model="adversarial")
    return -leakage
