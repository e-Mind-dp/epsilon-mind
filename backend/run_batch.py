# import os
# import pandas as pd
# from query_processor import process_query
# # from backend.experiments.query_processor import process_query


# def load_dataset(filename):
#     return pd.read_csv(filename)

# def ensure_folder(folder_path):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

# def main():
#     dataset_file = "backend/datasets/mental_health_diagnosis_treatment_.csv"
#     queries_file = "backend/experiments/dataset_queries/dataset_precise_queries_sample.csv"
#     output_folder = "backend/experiments/results"
#     output_file = os.path.join(output_folder, "healthcare_results_sample_timing.csv")

#     ensure_folder(output_folder)

#     df = load_dataset(dataset_file)

#     queries_df = pd.read_csv(queries_file)  # expects a column named "query"
#     queries = queries_df["query"].tolist()

#     past_queries = []
#     user_record = {
#         "role": "doctor",
#         "remaining_budget": 1000.0
#     }

#     results = []
#     for idx, query in enumerate(queries):
#         # try:
#         #     epsilon, true_ans, dp_ans = process_query(query, df, past_queries, user_record)
#         #     print(f"Query {idx}: Epsilon = {epsilon}")
#         try:
#             # (epsilon,
#             #  true_ans,
#             #  laplace_ans,
#             #  discrete_laplace_ans,
#             #  gaussian_ans,
#             #  discrete_gaussian_ans,
#             #  a1b1_ans,
#             #  a2b1_ans,
#             #  a3b1_ans,
#             #  a1b2_ans,
#             #  a2b2_ans,
#             #  a3b2_ans) = process_query(query, df, past_queries, user_record)
#             result = process_query(query, df, past_queries, user_record)

#             epsilon = result["epsilon"]
#             latency_total = result["latency_total"]
#             latency_llm = result["latency_llm"]
#             latency_rl = result["latency_rl"]
#             latency_perturb = result["latency_perturb"]
#             answers = result["answers"]

#             print(f"Query {idx}: Epsilon = {epsilon}")

#             # results.append({
#             #     "query_id": idx,
#             #     "query": query,
#             #     "epsilon": epsilon,
#             #     "true_answer": true_ans,
#             #     "dp_answer": dp_ans
#             # })
#             # results.append({
#             #     "query_id": idx,
#             #     "query": query,
#             #     "epsilon": epsilon,
#             #     "true_answer": true_ans,
#             #     "laplace_answer": laplace_ans,
#             #     "discrete_laplace_answer": discrete_laplace_ans,
#             #     "gaussian_answer": gaussian_ans,
#             #     "discrete_gaussian_answer": discrete_gaussian_ans,
#             #     "a1b1_result": a1b1_ans,
#             #     "a2b1_result": a2b1_ans,
#             #     "a3b1_result": a3b1_ans,
#             #     "a1b2_result": a1b2_ans,
#             #     "a2b2_result": a2b2_ans,
#             #     "a3b2_result": a3b2_ans
#             # })
#             results.append({
#                 "query_id": idx,
#                 "query": query,
#                 "epsilon": epsilon,
#                 "latency_total": latency_total,
#                 "latency_llm": latency_llm,
#                 "latency_rl": latency_rl,
#                 "latency_perturb": latency_perturb,
#                 "true_answer": answers["true"],
#                 "laplace_answer": answers["laplace"],
#                 "discrete_laplace_answer": answers["discrete_laplace"],
#                 "gaussian_answer": answers["gaussian"],
#                 "discrete_gaussian_answer": answers["discrete_gaussian"],
#                 "a1b1_result": answers["a1b1"],
#                 "a2b1_result": answers["a2b1"],
#                 "a3b1_result": answers["a3b1"],
#                 "a1b2_result": answers["a1b2"],
#                 "a2b2_result": answers["a2b2"],
#                 "a3b2_result": answers["a3b2"]
#             })

#         except Exception as e:
#             print(f"Query {idx} failed: {e}")
#             # results.append({
#             #     "query_id": idx,
#             #     "query": query,
#             #     "epsilon": None,
#             #     "true_answer": None,
#             #     "dp_answer": None,
#             #     "error": str(e)
#             # })
#             # results.append({
#             #     "query_id": idx,
#             #     "query": query,
#             #     "epsilon": None,
#             #     "true_answer": None,
#             #     "dp_answer": None,
#             #     "laplace_answer": None,
#             #     "discrete_laplace_answer": None,
#             #     "gaussian_answer": None,
#             #     "discrete_gaussian_answer": None,
#             #     "a1b1_result": None,
#             #     "a2b1_result": None,
#             #     "a3b1_result": None,
#             #     "a1b2_result": None,
#             #     "a2b2_result": None,
#             #     "a3b2_result": None,
#             #     "error": str(e)
#             # })
#             results.append({
#                 "query_id": idx,
#                 "query": query,
#                 "epsilon": None,
#                 "latency_total": None,
#                 "latency_llm": None,
#                 "latency_rl": None,
#                 "latency_perturb": None,
#                 "true_answer": None,
#                 "laplace_answer": None,
#                 "discrete_laplace_answer": None,
#                 "gaussian_answer": None,
#                 "discrete_gaussian_answer": None,
#                 "a1b1_result": None,
#                 "a2b1_result": None,
#                 "a3b1_result": None,
#                 "a1b2_result": None,
#                 "a2b2_result": None,
#                 "a3b2_result": None,
#                 "error": str(e)
#             })


#     results_df = pd.DataFrame(results)
#     results_df.to_csv(output_file, index=False)
#     print(f"Batch processing complete. Results saved to {output_file}")

# if __name__ == "__main__":
#     main()








import os
import pandas as pd
from query_processor import process_query

def load_dataset(filename):
    return pd.read_csv(filename)

def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def run_batch(domain, role, dataset_path, query_path, output_path):
    print(f"\nProcessing domain: {domain}, role: {role}")

    ensure_folder(os.path.dirname(output_path))

    df = load_dataset(dataset_path)
    queries_df = pd.read_csv(query_path)
    queries = queries_df["query"].tolist()

    user_record = {
        "role": role,
        "remaining_budget": 1000.0
    }

    past_queries = []
    results = []

    for idx, query in enumerate(queries):
        try:
            (epsilon,
             true_ans,
             laplace_ans,
             discrete_laplace_ans,
             gaussian_ans,
             discrete_gaussian_ans,
             a1b1_ans,
             a2b1_ans,
             a3b1_ans,
             a1b2_ans,
             a2b2_ans,
             a3b2_ans) = process_query(query, df, past_queries, user_record)

            print(f"[{domain} | {role}] Query {idx}: Epsilon = {epsilon}")

            results.append({
                "query_id": idx,
                "query": query,
                "epsilon": epsilon,
                "true_answer": true_ans,
                "laplace_answer": laplace_ans,
                "discrete_laplace_answer": discrete_laplace_ans,
                "gaussian_answer": gaussian_ans,
                "discrete_gaussian_answer": discrete_gaussian_ans,
                "a1b1_result": a1b1_ans,
                "a2b1_result": a2b1_ans,
                "a3b1_result": a3b1_ans,
                "a1b2_result": a1b2_ans,
                "a2b2_result": a2b2_ans,
                "a3b2_result": a3b2_ans
            })
        except Exception as e:
            print(f"[{domain} | {role}] Query {idx} failed: {e}")
            results.append({
                "query_id": idx,
                "query": query,
                "epsilon": None,
                "true_answer": None,
                "laplace_answer": None,
                "discrete_laplace_answer": None,
                "gaussian_answer": None,
                "discrete_gaussian_answer": None,
                "a1b1_result": None,
                "a2b1_result": None,
                "a3b1_result": None,
                "a1b2_result": None,
                "a2b2_result": None,
                "a3b2_result": None,
                "error": str(e)
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"✔️ Results saved to {output_path}")

def main():
    domains = {
        "healthcare": {
            "dataset": "backend/datasets/healthcare.csv",
            "queries": "backend/experiments/dataset_queries/healthcare_queries.csv"
        },
        "mobility": {
            "dataset": "backend/datasets/mobility.csv",
            "queries": "backend/experiments/dataset_queries/mobility_queries.csv"
        },
        "finance": {
            "dataset": "backend/datasets/finance.csv",
            "queries": "backend/experiments/dataset_queries/finance_queries.csv"
        },
        "smart_energy": {
            "dataset": "backend/datasets/smart_energy.csv",
            "queries": "backend/experiments/dataset_queries/smart_energy_queries.csv"
        }
    }

    roles = ["doctor", "public"]

    for domain, paths in domains.items():
        for role in roles:
            output_file = f"backend/experiments/results/{domain}_{role}_results_RL.csv"
            run_batch(domain, role, paths["dataset"], paths["queries"], output_file)

if __name__ == "__main__":
    main()
