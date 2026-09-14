import os

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# GET GROQ CONFIGURATION
# ==========================================

api_key = os.getenv("GROQ_API_KEY")

model_name = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b"
)


# ==========================================
# CHECK API KEY
# ==========================================

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Please check your .env file."
    )


# ==========================================
# CREATE GROQ CLIENT
# ==========================================

client = Groq(
    api_key=api_key
)


# ==========================================
# TEST GROQ CONNECTION
# ==========================================

def test_groq():

    print("\n" + "=" * 60)
    print("TESTING GROQ API CONNECTION")
    print("=" * 60)

    print(f"\nModel: {model_name}")
    print("\nGenerating retention strategy...")

    response = client.chat.completions.create(

    model=model_name,

    messages=[
        {
            "role": "user",
            "content": """
You are a professional telecom customer retention analyst.

Customer information:
- Churn Probability: 80%
- Contract: Month-to-month
- Tenure: 1 month
- Payment Method: Electronic check

Give exactly 3 short and practical retention recommendations.

Important:
- Use only the information provided.
- Do not invent complaints, customer behavior, budget problems,
  usage decline, dissatisfaction, or other facts.
- Give only the final recommendations.
"""
        }
    ],

    temperature=0.3,

    reasoning_effort="none",
    reasoning_format="hidden",

    max_completion_tokens=300
)

    result = response.choices[0].message.content

    print("\n" + "=" * 60)
    print("GROQ CONNECTION SUCCESSFUL!")
    print("=" * 60)

    print("\nAI RETENTION STRATEGY:\n")

    print(result)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    test_groq()