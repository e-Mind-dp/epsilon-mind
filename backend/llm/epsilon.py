import openai
from config import DEFAULT_MODEL
# from config import FALLBACK_MODEL


def decide_epsilon(query: str, sensitivity: str) -> float:
    prompt = f"""
You are a privacy AI. Given the following query and its sensitivity classification, assign a suitable privacy budget ε.
- Return a float between 0.1 and 1.2
- Lower ε means higher privacy
- Give only the numeric value
- If the sensitivity is "Extreme", then always return 0.0

Query: "{query}"
Sensitivity: {sensitivity}

Return only the epsilon value (e.g., 0.3)
"""
    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        # model=FALLBACK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return float(response.choices[0].message.content.strip())
