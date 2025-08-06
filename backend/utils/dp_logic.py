import re
from llm.privacy_mechanism import *

MECHANISM_MAP = {
    "laplace": laplace_mechanism,
    "gaussian": gaussian_mechanism,
    "discrete_laplace": discrete_laplace_mechanism,
    "discrete_gaussian": discrete_gaussian_mechanism
}

def apply_dp_tagged_answer(answer, epsilon, mechanism="laplace", delta=1e-5):
    """
    Applies a selected DP mechanism to all [DP]...[/DP] values in the answer.
    """
    pattern = r"\[DP\]([-+]?\d*\.\d+|\d+)\[/DP\]"

    def add_noise(match):
        original_value = float(match.group(1))
        mech_fn = MECHANISM_MAP[mechanism]
        if mechanism in ["gaussian", "discrete_gaussian"]:
            noisy_value = mech_fn(original_value, epsilon, delta=delta)
        else:
            noisy_value = mech_fn(original_value, epsilon)  
        return str(round(noisy_value, 2))

    return re.sub(pattern, add_noise, answer)
