# import re
# from llm.privacy_mechanism import apply_laplace_mechanism

# def apply_dp_tagged_answer(answer, epsilon):
#     """
#     Finds [DP]number[/DP] in the string, applies Laplace noise to the number,
#     and replaces the tag with the noisy value (rounded).
#     """
#     pattern = r"\[DP\]([-+]?\d*\.\d+|\d+)\[/DP\]"

#     def add_noise(match):
#         original_value = float(match.group(1))
#         noisy_value, _= apply_laplace_mechanism(original_value, epsilon)
#         return str(round(noisy_value, 2))

#     dp_answer = re.sub(pattern, add_noise, answer)
#     return dp_answer







# apply_dp_tagged_answer.py

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
        # noisy_value = mech_fn(original_value, epsilon, delta=delta)
        # Call mechanism with correct args depending on mechanism type
        if mechanism in ["gaussian", "discrete_gaussian"]:
            noisy_value = mech_fn(original_value, epsilon, delta=delta)
        else:
            noisy_value = mech_fn(original_value, epsilon)  # no delta for laplace, discrete_laplace
        return str(round(noisy_value, 2))

    return re.sub(pattern, add_noise, answer)
