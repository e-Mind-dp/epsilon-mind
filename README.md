# εMind: A Context-Aware and Adversarially Robust Framework for Modular Differential Privacy


## Overview

**εMind** is a novel, context-aware differential privacy framework that dynamically allocates privacy budgets (ε) using reinforcement learning and metaheuristic optimization. Using a Deep Double Q-Network (DDQN) agent informed by query sensitivity, user role, query type, and semantic similarity, εMind intelligently balances privacy and utility while defending against adversarial query strategies.

The system integrates:

- A **DDQN agent** with a multi-objective reward function combining utility maximization, privacy cost minimization, and redundancy penalties.
- A **hybrid Differential Evolution - Particle Swarm Optimization (DE-PSO)** metaheuristic to post-optimize ε allocations under Stackelberg-style adversarial simulations.
- A suite of **composite perturbation mechanisms** combining multiple noise distributions for enhanced accuracy and differential privacy guarantees.
- Transformer-based **LLM analysis** to extract sensitivity, query type and LLM confidence for adaptive decision-making.

εMind targets real-world scenarios in domains including healthcare, mobility, finance, and smart energy.

---

## Motivation

Differential privacy (DP) provides a mathematically rigorous framework for privacy-preserving data analysis. However, traditional DP mechanisms often rely on static, manually tuned ε values that fail to reflect the context of queries, user roles, and potential adversarial behaviors.

This static approach leads to suboptimal privacy-utility tradeoffs and vulnerability to sophisticated query attacks like fragmentation or composition attacks.

εMind addresses these challenges by:

- Contextualizing ε allocation with rich query and user metadata.
- Employing reinforcement learning to learn adaptive policies.
- Applying metaheuristic optimization for robustness against adversaries.
- Combining diverse noise perturbations in composite mechanisms for improved accuracy.

---

## Key Features and Contributions

- **Context-aware ε Allocation:** DDQN agent uses a 6-dimensional state vector (sensitivity, user role, query type, semantic similarity, remaining privacy budget, sensitivity confidence) to select ε from discrete actions.
- **Multi-Objective Reward:** Balances utility, privacy cost, and redundancy to guide RL training.
- **Hybrid DE-PSO Optimizer:** Fine-tunes ε vectors post-training by simulating adversarial query selection as a Stackelberg game, minimizing cumulative privacy leakage.
- **Composite Perturbation Functions:** Novel DP mechanisms defined by combining activation functions and base noise distributions, improving unbiasedness and accuracy.
- **Multi-Domain Support:** Demonstrated on healthcare, mobility, finance, and smart energy datasets.
- **Open Source Modular Design:** Well-structured Python codebase for extensibility and research reproducibility.

---

## System Architecture

εMind comprises four main components:

1. **Query Analyzer:** Uses Transformer-based LLM to analyze incoming queries, extracting sensitivity levels, query types, and confidence scores.
2. **DDQN RL Agent:** Receives query context and allocates ε budgets dynamically to balance privacy and utility.
3. **DE-PSO Metaheuristic Optimizer:** Runs Stackelberg-style adversarial simulations to optimize ε allocation vectors, improving resistance against privacy leakage attacks.
4. **Perturbation Module:** Applies composite perturbation mechanisms combining multiple noise distributions to satisfy differential privacy guarantees with minimal accuracy loss.

---

## Getting Started (VS Code Setup)

### 1. Install Required Packages

In your VS Code terminal, run:

### pip install -r requirements.txt

### 2. Start the Backend Server

In one terminal, navigate to the root directory and run:

### python backend/app.py

This will start the Flask API at:
http://127.0.0.1:5000/

### 3. Launch the Frontend (Streamlit UI)

In a second terminal, run:

### streamlit run app_ui.py

This will open the Streamlit-based interface in your browser.

### Make sure both terminals remain open during runtime. VS Code’s split-terminal feature is ideal for this workflow.


### NOTE
Please make a new file named config.py under the backend folder and add your own openai API details for complete access.

---

## Procedure
1. Once the SecuQuery page loads, register with an account and proceed to login with the same
2. Select the dataset type from the drop-dowm menu to ask queries
3. The dataset links have been attched below for reference (refer to that data to ask potential questions)
4. Some sample questions have been added for each dataset type under backend -> experiments -> dataset_queries, which can be used to ask questions as well
5. Put in the query into the box and submit it for the answer 

### Datasets
1. Healthcare: https://corgis-edu.github.io/corgis/csv/hospitals/
2. Mobility: https://www.google.com/covid19/mobility/index.html?hl=en
3. Finance: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
4. Smart energy: https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data



