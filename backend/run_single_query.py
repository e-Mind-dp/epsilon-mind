import torch
import pandas as pd
import json
import os

from rl_agent.agent import DDQNAgent
from rl_agent.utils import encode_state, get_epsilon_from_action
# from similarity import max_similarity
from utils.query_openai import generate_pandas_code
from utils.query_executor import execute_pandas_expression
from utils.dp_logic import apply_dp_tagged_answer
from utils.composite_mechanisms import *


# from de_pso_optimizer import DEPSOOptimizer
# from fitness_function import compute_fitness

# === CONFIGURABLE FLAGS ===
USE_TRANSFORMER = True
USE_DDQN = True
USE_DE_PSO = True
USE_ADVERSARIAL_SIM = True
USE_COMPOSITE_A3B1 = False
USE_LAPLACE_ONLY = True

dummy_user_record = {
    "role": "doctor",
    "remaining_budget": 1000.0
}

# === INPUT QUERY & DATA ===
query = "Which state has the highest average cost for Heart Failure?"
df = pd.read_csv("backend/datasets/healthcare.csv")

# === Step 1: Preprocess the query ===
if USE_TRANSFORMER:
    from llm.sensitivity import classify_sensitivity
    sensitivity_info = classify_sensitivity(query)
    sensitivity = sensitivity_info["sensitivity"]
    confidence = sensitivity_info["confidence"]
    query_type = sensitivity_info["query_type"]

else:
    sensitivity, confidence, query_type, role = "medium", 0.8, "aggregate", "doctor"

similarity = 0.5
remaining_budget = 1000.0

# === Step 2: Choose epsilon ===
if USE_LAPLACE_ONLY:
    epsilon = 1.5

elif USE_DDQN:
    # Load trained DDQN agent
    agent = DDQNAgent(input_dim=6, output_dim=20)
    agent.q_net.load_state_dict(torch.load("ddqn_q_network.pth"))
    agent.q_net.eval()
    
    state = encode_state(
        sensitivity, dummy_user_record["role"], query_type,
        similarity, remaining_budget, confidence
    )
    action = agent.select_action(state, epsilon=0.05)
    epsilon = get_epsilon_from_action(action)

else:
    epsilon = 1.0  # Fixed value for no-RL baseline

# # === Step 3: Apply DE–PSO optimization if enabled ===
# if USE_DE_PSO:
#     optimizer = DEPSOOptimizer(
#         num_agents=10, num_generations=30, num_queries=20,
#         use_adversary=USE_ADVERSARIAL_SIM
#     )
#     epsilon_vector = optimizer.run()
#     epsilon = float(epsilon_vector[0])  # Just for 1 query

# === Step 4: Compute the answer ===
pandas_code = generate_pandas_code(df, query)
true_answer = execute_pandas_expression(pandas_code, df)

if USE_LAPLACE_ONLY:
    # dp_answer = apply_laplace_mechanism(true_answer, epsilon)
    dp_answer = apply_dp_tagged_answer(true_answer, epsilon, mechanism="laplace")
else:
    # act_fn = "A3" if USE_COMPOSITE_A3B1 else "A2"
    # dp_answer = apply_composite_mechanism(true_answer, epsilon, act_fn, "B1")
    dp_answer = a3b1_result = apply_composite_mechanism(true_answer, epsilon, activation="A3", noise="B1", T=4.0, alpha= 5, beta=1, lower_bound=0, upper_bound=1000)

# === Step 5: Save output ===
ablation_name = ""
if USE_LAPLACE_ONLY:
    ablation_name = "laplace_only"
elif not USE_DE_PSO:
    ablation_name = "no_deso"
elif not USE_DDQN:
    ablation_name = "no_ddqn"
elif not USE_TRANSFORMER:
    ablation_name = "no_transformer"
elif not USE_ADVERSARIAL_SIM:
    ablation_name = "no_adversary"
elif not USE_COMPOSITE_A3B1:
    ablation_name = "a2b1"
else:
    ablation_name = "full_emind"

output = {
    "query": query,
    "true_answer": str(true_answer),
    "dp_answer": str(dp_answer),
    "epsilon": float(epsilon),
}

out_path = f"backend/experiments/manual_outputs/{ablation_name}.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"[✓] Saved result for {ablation_name} → {out_path}")
