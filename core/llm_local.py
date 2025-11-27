# core/llm_local.py
"""
Groq Llama-3 AI Explanation Engine
Cloud-ready, ultra fast, zero local model needed.
"""

import os
import requests

def generate_local_explanation(user_json, score, feats):
    """Generate explanation using Groq Llama API."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "LLM explanation is unavailable: GROQ_API_KEY not found."

    prompt = f"""
You are a senior financial credit analyst.
Analyze the user's credit behavior and provide:

1. A 3–4 sentence professional summary.
2. 3 clear recommendations to improve their financial health.

Eligibility Score: {score}

Key Features:
Repayment Rate: {feats.get('repayment_rate_12m')}
Bounce Count: {feats.get('bounce_count')}
Monthly Volatility: {feats.get('monthly_volatility')}
Late Payments: {feats.get('late_payments_12m')}
Loyalty Score: {feats.get('loyalty_score')}

Keep the tone formal, clear, and easy to understand.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert financial credit analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"LLM Error: {str(e)}"
