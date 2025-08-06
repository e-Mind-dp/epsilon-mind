import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ------------------------------
# Configurations
# ------------------------------
domains = ['healthcare', 'mobility', 'finance', 'smart_energy']
roles = ['doctor', 'public']

role_labels = {
    'doctor': 'Authorised User',
    'public': 'Unauthorised User'
}

dp_columns = [
    'laplace_answer', 'discrete_laplace_answer', 'gaussian_answer', 'discrete_gaussian_answer',
    'a1b1_result', 'a2b1_result', 'a3b1_result', 'a1b2_result', 'a2b2_result', 'a3b2_result'
]

legend_names = [
    'Lap', 'Dis_Lap', 'Gau', 'Dis_Gau',
    'A1B1', 'A2B1', 'A3B1', 'A1B2', 'A2B2', 'A3B2'
]

epsilon_bins = [0.1, 0.5, 1.0, 1.5, 2.0]


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

def prepare_boxplot_data(df, dp_col, epsilon_bins):
    df = df[df['true_answer'].apply(is_scalar)].copy()
    df['true_val'] = df['true_answer'].apply(extract_number)
    dp_val_col = dp_col + '_val'
    df[dp_val_col] = df[dp_col].apply(extract_number)
    df.dropna(subset=['true_val', dp_val_col, 'epsilon'], inplace=True)

    df['epsilon_bin'] = pd.cut(df['epsilon'], bins=epsilon_bins, include_lowest=True)
    df['epsilon_label'] = df['epsilon_bin'].apply(lambda x: round(x.right, 1))  # i+1 style
    df['rel_error'] = (abs(df[dp_val_col] - df['true_val']) / (abs(df['true_val']) + 1e-6)).clip(upper=10.0)
    return df[['epsilon_label', 'rel_error']]


# ------------------------------
# Plotting per Domain
# ------------------------------
plt.style.use("ggplot")
sns.set_context("notebook", font_scale=0.9)

for domain in domains:
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(27, 6), sharex=True)
    fig.suptitle(f'Accuracy Loss — {domain.title()}', fontsize=16)

    for row, role in enumerate(roles):
        file_path = f"backend/experiments/results/{domain}_{role}_results.csv"
        if not os.path.exists(file_path):
            print(f"[SKIP] File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        for col_idx, dp_mech in enumerate(dp_columns):
            data = prepare_boxplot_data(df, dp_mech, epsilon_bins)
            if data.empty:
                print(f"[SKIP] No valid data for {domain}-{role}-{dp_mech}")
                continue

            ax = axes[row, col_idx]
            sns.boxplot(
                x='epsilon_label',
                y='rel_error',
                data=data,
                ax=ax,
                linewidth=1,
                fliersize=2,
                boxprops=dict(alpha=0.8),
                medianprops=dict(linestyle=':', color='black')
            )

            ax.set_title(f"{legend_names[col_idx]}", fontsize=9)
            ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(f"{role_labels[role]}\nAccuracy Loss", fontsize=10)
            else:
                ax.set_ylabel("")

            ax.tick_params(axis='x', rotation=45)

            # Dynamic Y-axis using IQR
            q1 = data['rel_error'].quantile(0.25)
            q3 = data['rel_error'].quantile(0.75)
            iqr = q3 - q1
            upper_whisker = q3 + 1.5 * iqr
            ax.set_ylim(0, min(12, upper_whisker * 1.2))

            print(f"[{domain.upper()} - {role_labels[role].upper()} - {legend_names[col_idx]}] "
                  f"Avg RE: {data['rel_error'].mean():.3f}, "
                  f"Median RE: {data['rel_error'].median():.3f}, "
                  f"Bins: {data['epsilon_label'].nunique()}")
            

    # Set global X-label only once for the figure
    fig.text(0.5, 0.04, 'Privacy Budget', ha='center', fontsize=12)

    plt.subplots_adjust(top=0.88, bottom=0.15, wspace=1.2, hspace=0.3)  # more space between subplots
    plt.savefig(f"accuracy_loss_{domain}.png", dpi=300)
    plt.show()
