import numpy as np

class EpsilonStrategy:
    def __init__(self, base_epsilons=None, num_queries=20):
        self.num_queries = num_queries
        if base_epsilons is not None:
            self.epsilons = np.clip(base_epsilons, 0.1, 2.0)
        else:
            self.epsilons = np.random.uniform(0.1, 2.0, num_queries)

    def get_policy(self):
        return self.epsilons

    def set_policy(self, new_epsilons):
        self.epsilons = np.clip(new_epsilons, 0.1, 2.0)
