import numpy as np
from fitness_function import compute_fitness

class DEPSOOptimizer:
    def __init__(self, num_agents=10, num_generations=50, num_queries=20):
        self.num_agents = num_agents
        self.num_generations = num_generations
        self.num_queries = num_queries
        self.bounds = (0.1, 2.0)  # Epsilon values

        # Initialize agents and velocities
        self.agents = np.random.uniform(*self.bounds, (num_agents, num_queries))
        self.velocities = np.random.uniform(-0.5, 0.5, (num_agents, num_queries))

        # Initialize personal bests
        self.personal_best = self.agents.copy()
        self.personal_best_scores = np.array([compute_fitness(a) for a in self.agents])

        # Initialize global best
        self.global_best = self.personal_best[np.argmax(self.personal_best_scores)]

    def optimize(self):
        self.best_fitness_history = []

        for gen in range(self.num_generations):
            for i in range(self.num_agents):
                # --- DE mutation ---
                idxs = np.random.choice(np.delete(np.arange(self.num_agents), i), 3, replace=False)
                a, b, c = idxs
                F = 0.5  # DE scaling factor

                de_candidate = self.agents[a] + F * (self.agents[b] - self.agents[c])
                de_candidate = np.clip(de_candidate, *self.bounds)
                de_fitness = compute_fitness(de_candidate)

                # --- PSO update ---
                inertia = 0.5
                cognitive = 0.3
                social = 0.3

                velocity_update = (
                    inertia * self.velocities[i] +
                    cognitive * (self.personal_best[i] - self.agents[i]) +
                    social * (self.global_best - self.agents[i])
                )

                pso_candidate = self.agents[i] + velocity_update
                pso_candidate = np.clip(pso_candidate, *self.bounds)
                pso_fitness = compute_fitness(pso_candidate)

                # --- Choose better candidate ---
                if pso_fitness > de_fitness:
                    self.agents[i] = pso_candidate
                    self.velocities[i] = velocity_update
                    selected_fitness = pso_fitness
                else:
                    self.agents[i] = de_candidate
                    # DE doesn’t update velocity
                    selected_fitness = de_fitness

                # --- Update personal best if needed ---
                if selected_fitness > self.personal_best_scores[i]:
                    self.personal_best[i] = self.agents[i]
                    self.personal_best_scores[i] = selected_fitness

            # --- Update global best ---
            self.global_best = self.personal_best[np.argmax(self.personal_best_scores)]
            best_score = np.max(self.personal_best_scores)
            self.best_fitness_history.append(best_score)

            print(f"[GEN {gen+1}] Best Fitness: {best_score:.4f}")

        return self.global_best, self.best_fitness_history
