import numpy as np
from scipy.optimize import root_scalar

# -------------------------
# Activation Functions Ai
# -------------------------

def A1(q, alpha=5):
    return (1 / alpha) * np.log(q / (1 - q))

def A1_inv(z, alpha=5):
    exp_term = np.exp(alpha * z)
    return exp_term / (1 + exp_term)

def A1_derivative(q, alpha=5):
    return 1 / (alpha * q * (1 - q))

def A2(q, beta=1):
    return q - beta * (q - q ** 2)

def A2_inv(z, beta=1):
    # Solve z = q - β(q - q²) = q - βq + βq²
    # => βq² + (1 - β)q - z = 0
    a = beta
    b = 1 - beta
    c = -z
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return np.nan
    sqrt_d = np.sqrt(discriminant)
    q1 = (-b + sqrt_d) / (2*a)
    q2 = (-b - sqrt_d) / (2*a)
    return q1 if 0 <= q1 <= 1 else q2

def A2_derivative(q, beta=1):
    return 1 - beta + 2 * beta * q

def A3(q):
    return np.log(q / (1 - q))

def A3_inv(z):
    exp_term = np.exp(z)
    return exp_term / (1 + exp_term)

def A3_derivative(q):
    return 1 / (q * (1 - q))

# -------------------------
# Noise Functions Bj
# -------------------------

def sample_B1_truncated_logistic(T):
    while True:
        z = np.random.logistic(0, 1)
        if -T <= z <= T:
            return z

def sample_B2_reflected_exponential(lambda_, T):
    u = np.random.uniform(0, 1)
    direction = 1 if np.random.rand() < 0.5 else -1
    magnitude = -np.log(1 - u * (1 - np.exp(-lambda_ * T))) / lambda_
    return direction * magnitude
