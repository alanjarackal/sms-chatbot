import requests
import json

url = "http://localhost:8002/api/chat"
headers = {"Content-Type": "application/json"}

def test_query(message, description):
    print(f"\nTesting: {description}")
    payload = {"message": message, "client_id": "rajesh-123"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

test_query("What is the status of my land dispute?", "Case Specific Query")
test_query("What is the capital of India?", "General Knowledge Query")
