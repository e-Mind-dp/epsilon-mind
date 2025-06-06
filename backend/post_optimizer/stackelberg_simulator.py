# stackelberg_simulator.py

from fitness_function import simulate_user_response
from epsilon_strategy import EpsilonStrategy

class StackelbergSimulator:
    def __init__(self, epsilon_strategy):
        self.epsilon_strategy = epsilon_strategy

    def evaluate(self):
        policy = self.epsilon_strategy.get_policy()
        leakage = simulate_user_response(policy, user_model="adversarial")
        return leakage
