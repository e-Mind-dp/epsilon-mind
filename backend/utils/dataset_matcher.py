import openai

def match_dataset_name_llm(hint, all_datasets):
    prompt = f"""
You are a smart assistant. A user wants to query a dataset.

User's hint: "{hint}"

Here are the available datasets:
{', '.join(all_datasets)}

Which dataset name from above matches the user's intent best?

Only reply with the exact filename from the list above.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()