import openai
from config import DEFAULT_MODEL
# from config import FALLBACK_MODEL


def query_openai(df, user_query):
    # Prepare a trimmed-down view of the dataset for context
    context = df.head(10).to_csv(index=False)

    prompt = f"""
You are a data assistant with access to datasets. A user has asked the following query:

Query: "{user_query}"

Below is a preview of the dataset (first 10 rows):

{context}

Based on this data, respond to the query using insights from the dataset. If insufficient information is present, say "Not enough data".

Keep your response clear and data-informed, and to the point, do not show calculations or display the datasets at all.
Additionally, if your answer includes a numeric result that needs to be privacy-protected,
wrap only the final result value in [DP]...[/DP] tags.

Example: "The average sleep score is [DP]6.0[/DP]."
"""
    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        # model=FALLBACK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
