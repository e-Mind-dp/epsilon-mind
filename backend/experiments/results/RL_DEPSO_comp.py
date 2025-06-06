import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ------------------------------
# Config
# ------------------------------
# RL_FILE = "backend/experiments/results/healthcare_results_sample_rl.csv"
# OPT_FILE = "backend/experiments/results/healthcare_results_sample.csv"
RL_FILE = "backend/experiments/results/healthcare_public_results_RL.csv"
OPT_FILE = "backend/experiments/results/healthcare_public_results.csv"

MECHANISMS = [
    'laplace_answer', 'discrete_laplace_answer', 'gaussian_answer', 'discrete_gaussian_answer',
    'a1b1_result', 'a2b1_result', 'a3b1_result', 'a1b2_result', 'a2b2_result', 'a3b2_result'
]
LABELS = [
    'Lap', 'Dis_Lap', 'Gau', 'Dis_Gau',
    'A1B1', 'A2B1', 'A3B1', 'A1B2', 'A2B2', 'A3B2'
]
RMSE_CAP = 2000.0

# Colors: RL = Blue, OPT = Red (ggplot palette)
sns.set_theme(style="whitegrid", context="notebook", palette="muted")
COLORS = sns.color_palette("muted")

# ------------------------------
# Helpers
# ------------------------------
def extract_number(text):
    if not isinstance(text, str):
        return np.nan
    match = re.search(r"(-?\d+\.\d+|-?\d+)", text)
    return float(match.group(0)) if match else np.nan

def compute_avg_metrics(df, mechanism):
    df['true_val'] = df['true_answer'].apply(extract_number)
    df['dp_val'] = df[mechanism].apply(extract_number)
    df.dropna(subset=['true_val', 'dp_val'], inplace=True)

    t = df['true_val']
    d = df['dp_val']
    re = (abs(d - t) / (abs(t) + 1e-6)).clip(upper=10.0).mean()
    rmse = np.sqrt(((d - t) ** 2).clip(upper=RMSE_CAP).mean())
    return re, rmse

# ------------------------------
# Load and Compute
# ------------------------------
df_rl = pd.read_csv(RL_FILE)
df_opt = pd.read_csv(OPT_FILE)

re_rl_all, re_opt_all = [], []
rmse_rl_all, rmse_opt_all = [], []

print("=== Mechanism-wise Performance Comparison ===")
print(f"{'Mechanism':<10} | {'RE (RL)':>8} | {'RE (Opt)':>8} | {'ΔRE %':>8} || {'RMSE (RL)':>10} | {'RMSE (Opt)':>10} | {'ΔRMSE %':>9}")
print("-" * 75)

for i, mech in enumerate(MECHANISMS):
    re_rl, rmse_rl = compute_avg_metrics(df_rl.copy(), mech)
    re_opt, rmse_opt = compute_avg_metrics(df_opt.copy(), mech)

    re_rl_all.append(re_rl)
    re_opt_all.append(re_opt)
    rmse_rl_all.append(rmse_rl)
    rmse_opt_all.append(rmse_opt)

    delta_re = 100 * (re_rl - re_opt) / (re_rl + 1e-6)
    delta_rmse = 100 * (rmse_rl - rmse_opt) / (rmse_rl + 1e-6)

    print(f"{LABELS[i]:<10} | {re_rl:8.3f} | {re_opt:8.3f} | {delta_re:8.2f}% || {rmse_rl:10.3f} | {rmse_opt:10.3f} | {delta_rmse:9.2f}%")

# ------------------------------
# Plot with ggplot Style
# ------------------------------
x = np.arange(len(MECHANISMS))
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# RE Plot
axes[0].plot(x, re_rl_all, marker='o', linestyle='-', color=COLORS[0], label='RL ε')
axes[0].plot(x, re_opt_all, marker='s', linestyle='--', color=COLORS[1], label='Optimized ε')
axes[0].set_ylabel("RE", fontsize=12)
axes[0].set_title("Relative Error (RE) Comparison", fontsize=14)
axes[0].grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

# RMSE Plot
axes[1].plot(x, rmse_rl_all, marker='o', linestyle='-', color=COLORS[0], label='RL ε')
axes[1].plot(x, rmse_opt_all, marker='s', linestyle='--', color=COLORS[1], label='Optimized ε')
axes[1].set_ylabel("RMSE", fontsize=12)
axes[1].set_title("Root Mean Squared Error (RMSE) Comparison", fontsize=14)
axes[1].grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

# X-axis
axes[1].set_xticks(x)
axes[1].set_xticklabels(LABELS, rotation=45, ha='right')
axes[1].set_xlabel("Perturbation Mechanism", fontsize=12)

# Shared Legend
fig.legend(['RL ε', 'Optimized ε'], loc='upper center', ncol=2, fontsize='medium', frameon=False)

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ggplot_style_mechanism_comparison.png", dpi=300)
plt.show()
