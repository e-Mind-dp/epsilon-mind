# import re
# import numpy as np
# from utils.perturbation_core import *

# # Sensitivity computation (ΔA) for given activation
# def compute_sensitivity(activation_name, alpha=5, beta=1):
#     q_vals = np.linspace(0.01, 0.99, 1000)
#     if activation_name == "A1":
#         grads = A1_derivative(q_vals, alpha)
#     elif activation_name == "A2":
#         grads = A2_derivative(q_vals, beta)
#     elif activation_name == "A3":
#         grads = A3_derivative(q_vals)
#     else:
#         raise ValueError("Invalid activation name")
#     return np.max(np.abs(grads))

# # Normalize raw query answers into (0,1)
# def normalize_answer(raw_value, lower_bound=0, upper_bound=1000):
#     """
#     Normalize raw answer to (0,1) based on assumed bounds.
#     Clips raw_value to [lower_bound, upper_bound].
#     """
#     clipped = np.clip(raw_value, lower_bound, upper_bound)
#     norm = (clipped - lower_bound) / (upper_bound - lower_bound)
#     # Clamp to (0,1) to avoid boundary issues
#     return np.clip(norm, 1e-6, 1 - 1e-6)

# # Denormalize back to original scale
# def denormalize_answer(norm_value, lower_bound=0, upper_bound=1000):
#     return norm_value * (upper_bound - lower_bound) + lower_bound

# # Main composite mechanism
# def apply_composite_mechanism(raw_q, epsilon, activation="A1", noise="B1",
#                               T=4.0, alpha=5, beta=1,
#                               lower_bound=0, upper_bound=1000):
#     """
#     Apply composite DP mechanism to raw query answer.
#     raw_q: raw numerical answer (e.g., 73, 1500, etc.)
#     epsilon: privacy parameter
#     Normalizes raw_q to (0,1), applies perturbation, then denormalizes.
#     """
#     # Step 0: Normalize input
#     q = normalize_answer(raw_q, lower_bound=lower_bound, upper_bound=upper_bound)

#     # Step 1: Activation
#     if activation == "A1":
#         Aq = A1(q, alpha)
#         A_inv = lambda z: A1_inv(z, alpha)
#         delta_A = compute_sensitivity("A1", alpha=alpha)
#     elif activation == "A2":
#         Aq = A2(q, beta)
#         A_inv = lambda z: A2_inv(z, beta)
#         delta_A = compute_sensitivity("A2", beta=beta)
#     elif activation == "A3":
#         Aq = A3(q)
#         A_inv = A3_inv
#         delta_A = compute_sensitivity("A3")
#     else:
#         raise ValueError("Invalid activation name")

#     # Step 2: Noise
#     if noise == "B1":
#         Z = sample_B1_truncated_logistic(T)
#     elif noise == "B2":
#         lambda_ = epsilon / delta_A
#         Z = sample_B2_reflected_exponential(lambda_, T)
#     else:
#         raise ValueError("Invalid noise name")

#     # Step 3: Inversion
#     z_tilde = Aq + Z
#     q_noisy = A_inv(z_tilde)
#     q_noisy = np.clip(q_noisy, 0, 1)

#     # Step 4: Denormalize
#     noisy_raw = denormalize_answer(q_noisy, lower_bound=lower_bound, upper_bound=upper_bound)
#     return round(noisy_raw, 4)











import re
import numpy as np
from utils.perturbation_core import *

# Sensitivity computation (ΔA) for given activation
def compute_sensitivity(activation_name, alpha=5, beta=1):
    q_vals = np.linspace(0.01, 0.99, 1000)
    if activation_name == "A1":
        grads = A1_derivative(q_vals, alpha)
    elif activation_name == "A2":
        grads = A2_derivative(q_vals, beta)
    elif activation_name == "A3":
        grads = A3_derivative(q_vals)
    else:
        raise ValueError("Invalid activation name")
    return np.max(np.abs(grads))

# Normalize raw query answers into (0,1)
def normalize_answer(raw_value, lower_bound=0, upper_bound=1000):
    clipped = np.clip(raw_value, lower_bound, upper_bound)
    norm = (clipped - lower_bound) / (upper_bound - lower_bound)
    return np.clip(norm, 1e-6, 1 - 1e-6)

# Denormalize back to original scale
def denormalize_answer(norm_value, lower_bound=0, upper_bound=1000):
    return norm_value * (upper_bound - lower_bound) + lower_bound

# Main composite mechanism
def apply_composite_mechanism(raw_q, epsilon, activation="A1", noise="B1",
                              T=4.0, alpha=5, beta=1,
                              lower_bound=0, upper_bound=1000):
    """
    Apply composite DP mechanism to all numbers in a string query answer.
    Replaces each number with its DP-perturbed version.
    """
    # Pattern to match floats or ints in string
    pattern = r"([-+]?\d*\.\d+|\d+)"

    def perturb_match(match):
        try:
            raw_value = float(match.group(1))
            # Step 0: Normalize
            q = normalize_answer(raw_value, lower_bound, upper_bound)

            # Step 1: Activation
            if activation == "A1":
                Aq = A1(q, alpha)
                A_inv = lambda z: A1_inv(z, alpha)
                delta_A = compute_sensitivity("A1", alpha)
            elif activation == "A2":
                Aq = A2(q, beta)
                A_inv = lambda z: A2_inv(z, beta)
                delta_A = compute_sensitivity("A2", beta)
            elif activation == "A3":
                Aq = A3(q)
                A_inv = A3_inv
                delta_A = compute_sensitivity("A3")
            else:
                raise ValueError("Invalid activation")

            # Step 2: Noise
            if noise == "B1":
                Z = sample_B1_truncated_logistic(T)
            elif noise == "B2":
                lambda_ = epsilon / delta_A
                Z = sample_B2_reflected_exponential(lambda_, T)
            else:
                raise ValueError("Invalid noise")

            # Step 3: Inversion and clipping
            z_tilde = Aq + Z
            q_noisy = np.clip(A_inv(z_tilde), 0, 1)

            # Step 4: Denormalize
            noisy_raw = denormalize_answer(q_noisy, lower_bound, upper_bound)
            return str(round(noisy_raw, 4))
        except Exception as e:
            return match.group(1)  # fallback to original if error

    # Replace all numeric matches in the string
    return re.sub(pattern, perturb_match, str(raw_q))
