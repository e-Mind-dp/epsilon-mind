import openai
from config import DEFAULT_MODEL

def generate_pandas_code(df, user_query):
    """
    Use OpenAI to convert natural language query to a pandas expression.
    Only send column names to the LLM.
    """
    column_names = list(df.columns)
    
    prompt = f"""
You are a Python data analyst.

The dataset has the following columns:
{column_names}

Your task is to convert the user's query into a valid one-line pandas expression using this DataFrame called 'df'.

Query: "{user_query}"

Respond with only the pandas expression. Do not explain anything.

Examples:
- df['age'].mean()
- df[df['diagnosis'] == 'depression']['age'].mean()
- df['treatment'].value_counts()
"""

    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    # return response.choices[0].message.content.strip()
    raw_code = response.choices[0].message.content.strip()

    # === Strip any accidental formatting ===
    sanitized = raw_code.strip("`").replace("python", "").strip()
    return sanitized

