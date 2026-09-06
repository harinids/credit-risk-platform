import os

key = os.environ.get("GROQ_API_KEY", "")
print(f"Key length: {len(key)}")
print(f"Key starts with: {key[:10]}")
print(f"Key ends with: {key[-6:]}")

from groq import Groq
client = Groq(api_key=key)
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "say hi"}],
    max_tokens=50,
    reasoning_effort="low",
)
print(response.choices[0].message.content)