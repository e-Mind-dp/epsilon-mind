import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style

# Apply ggplot style
style.use('ggplot')
plt.rcParams.update({
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black'
})


# Load fitness history
fitness_history = np.loadtxt("post_optimizer/fitness_history.txt")
generations = np.arange(1, len(fitness_history) + 1)
leakage_history = -fitness_history  # Leakage is negative of fitness

# Print improvement summary
initial_leakage = leakage_history[0]
final_leakage = leakage_history[-1]

print("Leakage Reduction via Stackelberg DE–PSO")
print(f"• Initial Leakage  : {initial_leakage:.4f} ε")
print(f"• Final Leakage    : {final_leakage:.4f} ε")
print(f"• Leakage Reduced  : {initial_leakage - final_leakage:.4f} ε")
print(f"• Total Generations: {len(generations)}")

# Plot
plt.figure(figsize=(9, 5.5))
plt.plot(generations, leakage_history, marker='o', color='steelblue', linewidth=2)
plt.xlabel("Generation", fontsize=12)
plt.ylabel("Cumulative Leakage (ε)", fontsize=12)
# plt.title("Leakage Reduction Over Generations (DE–PSO with Stackelberg Adversarial Simulation)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("post_optimizer/de_pso_leakage_progress.png", dpi=300)
plt.show()



