# sea_lion_min.py
import os, re, requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SEA_LION_API_KEY")
BASE_URL = os.getenv("SEA_LION_BASE_URL", "https://api.sea-lion.ai/v1")
MODEL    = "aisingapore/Llama-SEA-LION-v3.5-8B-R"

def sea_lion_chat(messages, max_tokens=1200, temperature=0.7, top_p=0.9):
    if not API_KEY:
        raise RuntimeError("SEA_LION_API_KEY is missing. Add it to your .env. 🙃")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "chat_template_kwargs": {"thinking_mode": "on"},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    # strip hidden chain-of-thought if present
    return re.sub(r".*?</think>\s*", "", content, flags=re.DOTALL).strip()

if __name__ == "__main__":
    # Example: simple Q&A
    sys_prompt = {
        "role": "system",
        "content": "You are a concise medical assistant. Answer briefly and accurately."
    }
    user_msg = {"role": "user", "content": "I have a headache and mild fever. What should I do?"}
    print(sea_lion_chat([sys_prompt, user_msg]))
