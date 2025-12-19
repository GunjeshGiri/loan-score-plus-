import os
from groq import Groq

def run_llm_cloud(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior financial credit analyst. "
                    "Explain loan eligibility clearly for customers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content
