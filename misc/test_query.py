import requests

url = "http://localhost:5000/query"
payload = {
    "query": "What is the average sleep quality?",
    "dataset": "mental_health"
}

response = requests.post(url, json=payload)
print("Status:", response.status_code)
print("Response:", response.json())
