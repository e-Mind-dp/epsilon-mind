import os
import numpy as np
import json
import time
from epsilon_strategy import EpsilonStrategy
from de_pso_optimizer import DEPSOOptimizer
from stackelberg_simulator import StackelbergSimulator

if __name__ == "__main__":
    print("Starting Stackelberg DE–PSO Optimization...\n")

    strategy = EpsilonStrategy(num_queries=20)
    optimizer = DEPSOOptimizer(num_agents=20, num_generations=50, num_queries=20)
    
    start = time.time()
    best_epsilons, fitness_history = optimizer.optimize()
    strategy.set_policy(best_epsilons)

    simulator = StackelbergSimulator(strategy)
    leakage = simulator.evaluate()
    end = time.time()
    print(f"DE–PSO Optimization Time: {end - start:.2f} seconds")




    print("\n✅ Final Epsilon Strategy:", best_epsilons)
    print(f"🔐 Cumulative Privacy Leakage (Adversarial): {leakage:.4f}")
    print(f"🏆 Fitness Score: {-leakage:.4f}")

    print(f"📈 Fitness improved from {fitness_history[0]:.4f} to {fitness_history[-1]:.4f}")
    print(f"📉 Corresponding leakage reduced from {-fitness_history[0]:.4f} to {-fitness_history[-1]:.4f}")


    # At the end of your script, before saving:
    os.makedirs("post_optimizer", exist_ok=True)  # Create folder if it doesn't exist

    np.savetxt("post_optimizer/fitness_history.txt", fitness_history)

    # After optimization finishes
    with open("post_optimizer/best_epsilon_strategy.json", "w") as f:
        json.dump(best_epsilons.tolist(), f)
