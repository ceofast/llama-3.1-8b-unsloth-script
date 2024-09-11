import requests

url = "http://localhost:8000/generate"
data = {
    "instruction": "What is a famous tall tower in Paris?",
    "input": ""
}

response = requests.post(url, json=data)
print(response.json())
