from dotenv import load_dotenv
import os
import requests

load_dotenv()

key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={key}"

payload = {
    "content": {
        "parts": [{"text": "Hello world"}]
    }
}

r = requests.post(url, json=payload, timeout=30)

print("Status:", r.status_code)
print(r.text[:1000])