import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ------------------------------
# Configurations
# ------------------------------
domains = ['healthcare', 'mobility', 'finance', 'smart_energy']
roles = ['doctor', 'public']

dp_columns = [
    'laplace_answer', 'discrete_laplace_answer', 'gaussian_answer', 'discrete_gaussian_answer',
    'a1b1_result', 'a2b1_result', 'a3b1_result', 'a1b2_result', 'a2b2_result', 'a3b2_result'
]

legend_names = [
    'Lap', 'Dis_Lap', 'Gau', 'Dis_Gau',
    'A1B1', 'A2B1', 'A3B1', 'A1B2', 'A2B2', 'A3B2'
]

colors = plt.cm.tab10.colors  # 10 distinct colors

epsilon_bins = [0.1, 0.5, 1.0, 1.5, 2.0]
# x_labels = [f"{epsilon_bins[i]}–{epsilon_bins[i+1]}" for i in range(len(epsilon_bins) - 1)]
x_labels = [f"{epsilon_bins[i+1]}" for i in range(len(epsilon_bins) - 1)]
x_pos = np.arange(len(x_labels))

RMSE_CAP = 2000.0  # Cap RE to avoid bars being too small

# ------------------------------
# Helpers
# ------------------------------
def extract_number(text):
    if not isinstance(text, str):
        return np.nan
    match = re.search(r"(-?\d+\.\d+|-?\d+)", text)
    if match:
        return float(match.group(0))
    return np.nan

def is_scalar(text):
    return isinstance(text, str) and "['" not in text and '["' not in text

def compute_metrics(df, domain, role):
    df = df[df['true_answer'].apply(is_scalar)].copy()
    df['true_val'] = df['true_answer'].apply(extract_number)

    for col in dp_columns:
        df[col + '_val'] = df[col].apply(extract_number)

    required_cols = ['true_val', 'epsilon'] + [col + '_val' for col in dp_columns]
    df.dropna(subset=required_cols, inplace=True)

    df['epsilon_bin'] = pd.cut(df['epsilon'], bins=epsilon_bins, include_lowest=True)
    grouped = df.groupby('epsilon_bin', observed=False)

    metrics = {}

    for col in dp_columns:
        rel_errors, rmses = [], []
        print(f"\n[{domain.upper()} - {role.upper()}] Mechanism: {col}")
        for bin_interval, group in grouped:
            true_vals = group['true_val']
            dp_vals = group[col + '_val']
            if len(group) == 0:
                rel_errors.append(np.nan)
                rmses.append(np.nan)
                continue

            re = (abs(dp_vals - true_vals) / (abs(true_vals) + 1e-6)).clip(upper=10.0).mean()
            # mse = ((dp_vals - true_vals) ** 2).mean()
            mse = ((dp_vals - true_vals) ** 2).clip(upper=RMSE_CAP).mean()
            rmse = np.sqrt(mse)
            # rmse = min(rmse, RMSE_CAP)

            rel_errors.append(re)
            rmses.append(rmse)
            print(f"  Bin {bin_interval}: RE={re:.3f}, RMSE={rmse:.3f}, Count={len(group)}")

        metrics[col] = {'rel_error': rel_errors, 'rmse': rmses}
    return metrics

# ------------------------------
# Plotting
# ------------------------------
# fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 18), sharex=True, sharey=True,
#                          gridspec_kw={'hspace': 0.3, 'wspace': 0.2})

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 18), sharex=True, sharey=False,
                         gridspec_kw={'hspace': 0.3, 'wspace': 0.2})


for i, domain in enumerate(domains):
    for j, role in enumerate(roles):
        file_path = f"backend/experiments/results/{domain}_{role}_results.csv"
        ax1 = axes[i, j]
        ax2 = ax1.twinx()

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        metrics = compute_metrics(df, domain, role)

        bar_width = 0.08
        for k, mech in enumerate(dp_columns):
            offset = (k - len(dp_columns)/2) * bar_width + bar_width/2
            ax1.bar(x_pos + offset, metrics[mech]['rel_error'], width=bar_width,
                    color=colors[k % 10], alpha=0.7)

            ax2.plot(x_pos, metrics[mech]['rmse'], marker='o',
                     color=colors[k % 10], linestyle='--')

        ax1.set_title(f"{domain.replace('_', ' ').title()}", fontsize=11)
        if i == 3:
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        else:
            ax1.set_xticklabels([])

        # if j == 0:
        #     ax1.set_ylabel("RE", fontsize=10)
        # if j == 1:
        #     ax2.set_ylabel("RMSE", fontsize=10)

        ax1.set_ylabel("RE", fontsize=10)
        ax2.set_ylabel("RMSE", fontsize=10)

        # Dynamic y-axis limits for clarity
        all_re = []
        all_rmse = []
        for mech in dp_columns:
            all_re.extend([v for v in metrics[mech]['rel_error'] if not np.isnan(v)])
            all_rmse.extend([v for v in metrics[mech]['rmse'] if not np.isnan(v)])
        
        if all_re:
            ax1.set_ylim(0, min(12, max(all_re)*1.2))  # Slight padding
        if all_rmse:
            ax2.set_ylim(0, min(RMSE_CAP, max(all_rmse)*1.2))


# ------------------------------
# Shared Legend (Mechanism Name Only)
# ------------------------------
legend_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(legend_names))]
fig.legend(legend_lines, legend_names, loc='upper center', ncol=10, fontsize='medium', frameon=False)

# ------------------------------
# Save and Show
# ------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("privacy_utility_2x4_grid_final.png", dpi=300)
plt.show()







