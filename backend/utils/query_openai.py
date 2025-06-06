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




# def generate_pandas_code(df, expression):
#     """
#     Safely evaluate a pandas expression like 'df["Age"].mean()' and
#     return a stringified result suitable for LLM-style answers.
#     """
#     import pandas as pd

#     # Basic safety check to reject dangerous expressions
#     if "__" in expression or ";" in expression or "import" in expression:
#         return "Unsafe expression rejected."

#     # Clean up expression string if it has code formatting
#     expression = expression.strip("`").replace("python", "").strip()

#     # Allowed variables in eval
#     local_vars = {"df": df, "pd": pd}
#     try:
#         print(f"Evaluating expression:\n{expression}")  # Debug log

#         result = eval(expression, {"__builtins__": None}, local_vars)

#         # Check if result is None
#         if result is None:
#             return "Query returned no result."

#         # Handle Series result (e.g. df['col'].value_counts())
#         if isinstance(result, pd.Series):
#             if result.empty:
#                 return "Query returned empty result."
#             result = result.round(2).to_dict()
#             return " ".join([f"{k}: [DP]{v}[/DP]" for k, v in result.items()])

#         # Handle DataFrame result
#         if isinstance(result, pd.DataFrame):
#             if result.empty:
#                 return "Query returned empty result."
#             return "Result is a table (not supported for DP protection)."

#         # Handle numeric result
#         if isinstance(result, (int, float)):
#             return f"The result is [DP]{round(result, 2)}[/DP]."

#         # Fallback: convert to string
#         return str(result)

#     except Exception as e:
#         return f"Error executing query: {str(e)}"
