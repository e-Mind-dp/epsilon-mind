import pandas as pd

def summarize_results(file_path):
    df = pd.read_csv(file_path)

    # Columns to summarize
    latency_cols = [
        "latency_total",
        "latency_llm",
        "latency_rl",
        "latency_perturb"
    ]

    epsilon_col = ["epsilon"]

    numeric_cols = [
        "laplace_answer",
        "discrete_laplace_answer",
        "gaussian_answer",
        "discrete_gaussian_answer",
        "a1b1_result",
        "a2b1_result",
        "a3b1_result",
        "a1b2_result",
        "a2b2_result",
        "a3b2_result"
    ]

    # Convert answer columns (e.g., """The result is 12.34.""") to floats
    def extract_float(value):
        if pd.isna(value):
            return None
        try:
            stripped = str(value).replace("[DP]", "").replace("[/DP]", "")
            num = ''.join(c for c in stripped if c.isdigit() or c in ".-")
            return float(num)
        except:
            return None

    for col in numeric_cols:
        df[col] = df[col].apply(extract_float)

    summary = df[epsilon_col + latency_cols + numeric_cols].mean(numeric_only=True)
    print("===== AVERAGE METRICS =====")
    print(summary.round(4))

if __name__ == "__main__":
    file_path = "backend/experiments/results/healthcare_results_sample_timing.csv"
    summarize_results(file_path)
