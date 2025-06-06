import torch
import json
import pandas as pd
import time


from llm.sensitivity import classify_sensitivity
from rl_agent.utils import encode_state, optimised_epsilon, get_epsilon_from_action
from rl_agent.agent import DDQNAgent
from utils.query_openai import generate_pandas_code
from utils.query_executor import execute_pandas_expression
from utils.dp_logic import apply_dp_tagged_answer
from similarity import max_similarity_against_history
from utils.composite_mechanisms import *

# Load RL agent once globally
agent = DDQNAgent(input_dim=6, output_dim=20)
agent.q_net.load_state_dict(torch.load("ddqn_q_network.pth"))
agent.q_net.eval()

# Dummy user record with large budget for batch
dummy_user_record = {
    "role": "doctor",
    "remaining_budget": 1000.0
}

def process_query(query: str, df: pd.DataFrame, past_queries: list, user_record: dict = dummy_user_record):
    """
    Process a single query: get epsilon via RL, true answer, DP noisy answer.
    
    Args:
      - query: str, user query text
      - df: pd.DataFrame, dataset on which query runs
      - past_queries: list of past queries (str) for similarity calc
      - user_record: dict with keys role and remaining_budget
    
    
    """
    start_full = time.time()


    # Step 1: Sensitivity classification
    start = time.time()
    sensitivity_info = classify_sensitivity(query)
    end = time.time()
    llm_time = end - start


    sensitivity = sensitivity_info["sensitivity"]
    confidence = sensitivity_info["confidence"]
    query_type = sensitivity_info["query_type"]

    
    if sensitivity.strip().lower() == "extreme":
        raise ValueError("Query involves extreme sensitivity")

    # Step 2: Compute max similarity against past queries
    max_similarity = max_similarity_against_history(query, past_queries)

    role_int_to_str = {0: "doctor", 1: "researcher", 2: "public"}

    if isinstance(user_record["role"], int):
        user_record["role"] = role_int_to_str.get(user_record["role"], "public")

    print(f"Sensitivity info: {sensitivity_info}")
    # print(f"Max similarity: {max_similarity}")



    # Step 3: Encode state for RL agent
    state = encode_state(
        sensitivity=sensitivity,
        user_role=user_record["role"],
        query_type=query_type,
        similarity=max_similarity,
        budget=user_record["remaining_budget"],
        confidence=confidence
    )

    # print("Encoded state vector:", state)


    # Step 4: Use RL agent to select epsilon
    with torch.no_grad():
        start = time.time()
        action_idx = agent.select_action(state, epsilon=1.0)  # greedy
    # epsilon = optimised_epsilon(action_idx)
    epsilon = get_epsilon_from_action(action_idx)
    end = time.time()
    rl_time = end - start



    # Step 5: Get true answer
    expression = generate_pandas_code(df, query)
    if not expression or "df" not in expression:
        raise ValueError("Failed to interpret query")

    true_answer = execute_pandas_expression(df, expression)

    start = time.time()
    # Step 6: Apply DP noise with epsilon - baseline
    # dp_answer = apply_dp_tagged_answer(true_answer, epsilon)
    laplace_answer = apply_dp_tagged_answer(true_answer, epsilon, mechanism="laplace")
    discrete_laplace_answer = apply_dp_tagged_answer(true_answer, epsilon, mechanism="discrete_laplace")
    gaussian_answer = apply_dp_tagged_answer(true_answer, epsilon, mechanism="gaussian")
    discrete_gaussian_answer = apply_dp_tagged_answer(true_answer, epsilon, mechanism="discrete_gaussian")

    # Step 7: Apply DP noise with epsilon - perturbation
    a1b1_result = apply_composite_mechanism(true_answer, epsilon, activation="A1", noise="B1", T=4.0, alpha= 5, beta=1, lower_bound=0, upper_bound=1000)
    a2b1_result = apply_composite_mechanism(true_answer, epsilon, activation="A2", noise="B1", T=4.0, alpha= 5, beta=1, lower_bound=0, upper_bound=1000)
    a3b1_result = apply_composite_mechanism(true_answer, epsilon, activation="A3", noise="B1", T=4.0, alpha= 5, beta=1, lower_bound=0, upper_bound=1000)
    a1b2_result = apply_composite_mechanism(true_answer, epsilon, activation="A1", noise="B2", T=4.0, alpha= 5, beta=1, lower_bound=0, upper_bound=1000)
    a2b2_result = apply_composite_mechanism(true_answer, epsilon, activation="A2", noise="B2", T=4.0, alpha= 5, beta=1, lower_bound=0, upper_bound=1000)
    a3b2_result = apply_composite_mechanism(true_answer, epsilon, activation="A3", noise="B2", T=4.0, alpha= 5, beta=1, lower_bound=0, upper_bound=1000)

    end = time.time()
    perturbation_time = end - start



    # Convert answers to JSON strings for CSV serialization
    def serialize_answer(ans):
        try:
            return json.dumps(ans)
        except Exception:
            return str(ans)

    true_answer_str = serialize_answer(true_answer)
    # dp_answer_str = serialize_answer(dp_answer)
    
    laplace_answer_str = serialize_answer(laplace_answer)
    discrete_laplace_answer_str = serialize_answer(discrete_laplace_answer)
    gaussian_answer_str = serialize_answer(gaussian_answer)
    discrete_gaussian_answer_str = serialize_answer(discrete_gaussian_answer)

    a1b1_result_str = serialize_answer(a1b1_result)
    a2b1_result_str = serialize_answer(a2b1_result)
    a3b1_result_str = serialize_answer(a3b1_result)
    a1b2_result_str = serialize_answer(a1b2_result)
    a2b2_result_str = serialize_answer(a2b2_result)
    a3b2_result_str = serialize_answer(a3b2_result)
    

    end_full = time.time()
    total_latency = end_full - start_full







    # Step 8: Update past queries list
    past_queries.append(query)
    # print(f"ε = {epsilon:.4f} | Total = {total_latency:.3f}s | LLM = {llm_time:.3f}s | RL = {rl_time:.3f}s | Perturb = {perturbation_time:.3f}s")


    return epsilon, true_answer_str, laplace_answer_str, discrete_laplace_answer_str, gaussian_answer_str, discrete_gaussian_answer_str, a1b1_result_str, a2b1_result_str, a3b1_result_str, a1b2_result_str, a2b2_result_str, a3b2_result_str
#     return {
#     "epsilon": epsilon,
#     "latency_total": total_latency,
#     "latency_llm": llm_time,
#     "latency_rl": rl_time,
#     "latency_perturb": perturbation_time,
#     "answers": {
#         "true": true_answer_str,
#         "laplace": laplace_answer_str,
#         "discrete_laplace": discrete_laplace_answer_str,
#         "gaussian": gaussian_answer_str,
#         "discrete_gaussian": discrete_gaussian_answer_str,
#         "a1b1": a1b1_result_str,
#         "a2b1": a2b1_result_str,
#         "a3b1": a3b1_result_str,
#         "a1b2": a1b2_result_str,
#         "a2b2": a2b2_result_str,
#         "a3b2": a3b2_result_str
#     }
# }




