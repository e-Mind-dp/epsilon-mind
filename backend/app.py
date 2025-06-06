from flask import Flask, request, jsonify, session
import os
import pandas as pd
import bcrypt
import torch
import numpy as np


from utils.dataset_matcher import match_dataset_name_llm
from utils.query_openai_old import query_openai
from utils.query_openai import generate_pandas_code
from utils.query_executor import execute_pandas_expression
from llm.sensitivity import classify_sensitivity
from llm.epsilon import decide_epsilon
from llm.feedback import generate_privacy_feedback
from utils.dp_logic import apply_dp_tagged_answer
from user_db import get_user_by_email, register_user, get_user_record, update_user_record, store_user_query, get_all_user_queries
from budget import can_process_query, deduct_budget
from rl_agent.q_network import DQN
from rl_agent.utils import encode_state, get_epsilon_from_action, optimised_epsilon
from rl_agent.agent import DDQNAgent
from similarity import max_similarity_against_history




app = Flask(__name__)
DATASET_FOLDER = os.path.join("backend", "datasets")


# === Load trained RL agent once at startup ===
agent = DDQNAgent(input_dim=6, output_dim=20)  
agent.q_net.load_state_dict(torch.load("ddqn_q_network.pth"))
agent.q_net.eval()



def list_available_datasets():
    return [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]

def load_dataset(filename):
    filepath = os.path.join(DATASET_FOLDER, filename)
    return pd.read_csv(filepath)

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")
    dataset_hint = data.get("dataset")
    user_id = data.get("user_id")

    if not all([query, dataset_hint, user_id]):
        return jsonify({"error": "Query, dataset, and user ID are required"}), 400

    try:

        # === Privacy Budget Check ===
        user_record = get_user_record(user_id)

        if user_record is None:
            return jsonify({"error": "User not found"}), 404


        # Step 1: Get all available dataset filenames
        available_datasets = list_available_datasets()

        # Step 2: Use LLM to match hint to actual filename
        matched_dataset_filename = match_dataset_name_llm(dataset_hint, available_datasets)
        print("The chosen dataset is ", matched_dataset_filename)

        # Step 3: Load the matched dataset into a DataFrame
        df = load_dataset(matched_dataset_filename)

        # Step 4: Run sensitivity analysis
        sensitivity = classify_sensitivity(query)
        print("The sensitivity is ", sensitivity)

        # Normalize string for robust comparison
        if sensitivity["sensitivity"].strip().lower() == "extreme":
            return jsonify({
                "error": "This query involves sensitive information and cannot be processed."
            }), 400


        # === Compute max similarity with past queries ===
        past_queries = get_all_user_queries(user_id)
        max_similarity = max_similarity_against_history(query, past_queries)
        print("Max similarity with past queries: ", max_similarity)
        

        # === RL-based Epsilon Decision ===

        # 1. Construct the state vector
        input_sensitivity = sensitivity["sensitivity"]   
        input_remaining_budget = user_record["remaining_budget"]
        input_query_type = sensitivity["query_type"]  
        input_semantic_similarity = max_similarity
        input_user_role = user_record["role"]
        input_confidence = sensitivity["confidence"]
        
        # === RL: Encode state and predict epsilon ===
        state = encode_state(
            sensitivity=input_sensitivity,
            user_role=input_user_role,
            query_type=input_query_type,
            similarity=input_semantic_similarity,
            budget=input_remaining_budget,
            confidence=input_confidence
        )


        with torch.no_grad():
            action_idx = agent.select_action(state, epsilon=0.0)  # greedy selection
            print(f"Selected action index: {action_idx}")

        # epsilon = get_epsilon_from_action(action_idx)
        epsilon = optimised_epsilon(action_idx)
        # epsilon = get_epsilon_from_action(action_idx)
        


        print("The epsilon decided by RL model is:", epsilon)

        
        # Check if enough budget exists
        allowed, updated_record = can_process_query(user_record, epsilon)
        if not allowed:
            return jsonify({"error": "Privacy budget exhausted. Try again after reset."}), 400
        

        store_user_query(
            user_id = user_id,
            query_text = query,
            epsilon_used = epsilon,
            sensitivity = sensitivity["sensitivity"]
)

        
        # Step 7: Answer query with LLM using dataset
        answer_old = query_openai(df, query)
        print("OLDDD Answer to the query is ", answer_old)

        expression = generate_pandas_code(df, query)
        if not expression or "df" not in expression:
            return jsonify({"error": "Failed to interpret query."}), 400

        print("Generated expression:", expression)

        answer = execute_pandas_expression(df, expression)
        print("NEWWW Answer is ", answer)

        dp_answer = apply_dp_tagged_answer(answer, epsilon)
        print("Answer with DP is ", dp_answer)

        # Deduct and update budget
        updated_record = deduct_budget(updated_record, epsilon)
        update_user_record(user_id, updated_record)

        # Step 7: Generate feedback message
        feedback = generate_privacy_feedback(sensitivity, dp_answer)

        return jsonify({"feedback": feedback})


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    




    
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    existing_user = get_user_by_email(email)
    if existing_user:
        return jsonify({"error": "User already exists"}), 409

    result = register_user(email, password)
    if result:
        return jsonify({"message": "User registered successfully"}), 201
    return jsonify({"error": "Registration failed"}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = get_user_by_email(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({"message": "Login successful", "user_id": user['id']}), 200
    else:
        return jsonify({"error": "Invalid password"}), 401



if __name__ == "__main__":
    app.run(debug=True)
