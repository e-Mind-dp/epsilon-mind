import numpy as np

def laplace_mechanism(value, epsilon, sensitivity=1.0):
    noise = np.random.laplace(0.0, sensitivity / epsilon)
    return max(0, value + noise)

def gaussian_mechanism(value, epsilon, delta=1e-5, sensitivity=1.0):
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / epsilon)
    noise = np.random.normal(0.0, sigma)
    return max(0, value + noise)

def discrete_laplace_mechanism(value, epsilon, sensitivity=1.0):
    b = epsilon / sensitivity
    u = np.random.uniform(-0.5, 0.5)
    sign = 1 if u >= 0 else -1
    noise = sign * np.floor(np.log(1 - 2 * abs(u)) / np.log(1 - np.exp(-b)))
    return max(0, value + noise)

def discrete_gaussian_mechanism(value, epsilon, delta=1e-5, sensitivity=1.0):
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / epsilon)
    noise = int(np.round(np.random.normal(0.0, sigma)))
    return max(0, value + noise)