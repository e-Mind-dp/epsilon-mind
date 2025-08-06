import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load DE–PSO best policy
# --------------------------
with open("best_epsilon_strategy.json", "r") as f:
    epsilon_policy = np.array(json.load(f))

# Print basic policy stats
print("=== Epsilon Policy Statistics ===")
print(f"Min: {np.min(epsilon_policy):.4f}")
print(f"Max: {np.max(epsilon_policy):.4f}")
print(f"Mean: {np.mean(epsilon_policy):.4f}")
print(f"Top 5 Indices: {np.argsort(epsilon_policy)[-5:][::-1]}")
print(f"Top 5 Values: {np.sort(epsilon_policy)[-5:][::-1]}")
print("="*40)

# --------------------------
# Define attacker strategies
# --------------------------
def attacker_indices(policy, attacker):
    if attacker == "top5":
        return np.argsort(policy)[-5:]
    elif attacker == "top5_noisy":
        noise = np.random.normal(0, 0.05, size=len(policy))
        return np.argsort(policy + noise)[-5:]
    elif attacker == "middle5":
        sorted_idx = np.argsort(policy)
        mid_start = len(policy) // 2 - 2
        return sorted_idx[mid_start:mid_start + 5]
    elif attacker == "random5":
        return np.random.choice(len(policy), size=5, replace=False)
    else:
        raise ValueError("Unknown attacker model")

# --------------------------
# Run 100 simulations per attacker
# --------------------------
attackers = ['top5', 'top5_noisy', 'middle5', 'random5']
# attacker_labels = ['Top-5', 'Top-5 (Noisy)', 'Middle-5', 'Random-5']
attacker_labels = ['Greedy Max Exploiter', 'Noisy Estimator', 'Mid-range Selector', 'Uniform Sampler']


heatmap_matrix = np.zeros((len(attackers), len(epsilon_policy)))

np.random.seed(42)

for i, attacker in enumerate(attackers):
    for _ in range(100):
        idx = attacker_indices(epsilon_policy, attacker)
        heatmap_matrix[i, idx] += 1

# Print attacker targeting summary
print("=== Attacker Simulation Summary ===")
for i, attacker in enumerate(attackers):
    counts = heatmap_matrix[i]
    top_hit_indices = np.argsort(counts)[-5:][::-1]
    print(f"{attacker_labels[i]}:")
    for idx in top_hit_indices:
        print(f"  Index {idx} hit {int(counts[idx])} times")
    print("-" * 30)

# Print overall most targeted budgets
total_hits = np.sum(heatmap_matrix, axis=0)
top_total_hits = np.argsort(total_hits)[-5:][::-1]
print("=== Overall Most Targeted Privacy Budgets ===")
for idx in top_total_hits:
    print(f"Privacy Budget {idx}: selected {int(total_hits[idx])} times")
print("="*40)

# --------------------------
# Apply ggplot-style theme
# --------------------------
sns.set_theme(style="whitegrid")
sns.set_palette("muted")

plt.rcParams.update({
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.8,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': 'DejaVu Sans',
})

# --------------------------
# Plot Heatmap
# --------------------------
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    cbar_kws={'label': 'Selection Frequency'},
    xticklabels=[f"{i}" for i in range(len(epsilon_policy))],
    yticklabels=attacker_labels,
    linewidths=0.5,
    linecolor='lightgray',
    ax=ax
)

# ax.set_title("Privacy Budget Target Frequency Across Attacker Strategies", fontsize=16, weight='bold')
ax.set_xlabel("Privacy Budget")
ax.set_ylabel("Attacker Model")

plt.tight_layout()
plt.savefig("post_optimizer/attacker_focus_heatmap_ggstyle.png", dpi=300)
plt.show()

