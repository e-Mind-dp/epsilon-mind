import openai
from config import DEFAULT_MODEL

# Your sensitivity table (shortened for clarity, you can expand)
PHI_CATEGORIES = {
    "Critical": [
        "Name", "Social Security No.", "SSN", "Credit Card", "Debit Card",
        "National Identifiers", "Financial records", "Biometric data"
    ],
    "Sensitive": [
        "Email", "Date of Birth", "Phone Number", "Social Media Profiles",
        "Medical Data", "Home Address"
    ],
    "Quasi": [
        "Organization", "Zipcode", "IP Address", "MAC Address",
        "Ethnicity", "Gender"
    ]
}

def classify_sensitivity(query: str) -> dict:
    # Prompt to get sensitivity + extract phi-like entities
    prompt = f"""
You are a data privacy expert. Given the following user query:

"{query}"

1. Classify its **sensitivity** into one of: Low, Medium, High, Extreme:
    - Low: No sensitive entities
    - Medium: Only quasi identifiers or one sensitive term
    - High: Multiple sensitive terms or any critical term
    - Extreme: Multiple critical terms or exact identifiers of people

2. Identify which specific terms from the PHI_CATEGORIES given above are mentioned or implied in the query. Do not respond with category names like "Critical" or "Quasi" — only include all the exact terms relevant.

3. Identify the **query type** as one of:
    - individual: pertains to a specific person or record
    - aggregate: computes statistics or summaries across multiple records
    - filtering: applies filters (e.g., age > 60, gender = female)
    - temporal: involves time-based trends, changes, or durations
    - comparative: compares groups, attributes, or timelines
    - descriptive: non-personal information
    - unknown: cannot determine from the query

4. Estimate your **confidence** in your sensitivity classification (range: 0.1 to 1.0), where:
    - 0.1 means low certainty
    - 1.0 means extremely certain

Output strictly like this AND THEY CANNOT BE NULL/EMPTY:

Sensitivity: <Low/Medium/High/Extreme>
PHI_Categories: <comma-separated exact terms from the above list — do not write 'Critical' or 'Quasi'>
Query_Type: <aggregate/individual/update/delete/unknown>
Confidence: <one value between 0.1 and 1.0>  


Do not output any other text or explanation. This format is mandatory.

"""

    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    raw_output = response.choices[0].message.content.strip()
    # print("LLM raw output:\n", raw_output) 
    lines = raw_output.splitlines()

    result = {
        "sensitivity": None,
        "phi_categories" : None,
        "query_type": None,
        "confidence": 0.7  # Default fallback
    }

    for line in lines:
        line = line.strip()

        if line.startswith("Sensitivity:"):
            try:
                result["sensitivity"] = line.split("Sensitivity:")[1].strip()
            except IndexError:
                result["sensitivity"] = "Unknown"

        elif line.startswith("PHI_Categories:"):
            try:
                terms = line.split("PHI_Categories:")[1].strip(" []")
                result["phi_categories"] = [term.strip() for term in terms.split(",") if term.strip()]
            except IndexError:
                result["phi_categories"] = []

        elif line.startswith("Query_Type:"):
            try:
                result["query_type"] = line.split("Query_Type:")[1].strip()
            except IndexError:
                result["query_type"] = "unknown"

        elif line.startswith("Confidence:"):
            try:
                conf = float(line.split("Confidence:")[1].strip())
                result["confidence"] = max(0.1, min(1.0, conf))  # Clamp between 0.1 and 1.0
            except:
                pass  # Leave default if parsing fails

    return result