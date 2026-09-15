import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

EXTRACTION_SYSTEM_PROMPT = """
You are a precise data extraction assistant for a Wine Quality Prediction system.
Your job is to parse natural language user descriptions and extract key wine features into a raw, valid JSON object.

Extract the following features (use numbers only):
- "wine_type": "red" or "white" (default to "white" if unspecified)
- "fixed acidity": float or null
- "volatile acidity": float or null
- "citric acid": float or null
- "residual sugar": float or null
- "chlorides": float or null
- "free sulfur dioxide": float or null
- "total sulfur dioxide": float or null
- "density": float or null
- "pH": float or null
- "sulphates": float or null
- "alcohol": float or null

Rules:
1. Return ONLY a single raw valid JSON object. Do NOT wrap in markdown fences (```json ... ```) or add commentary.
2. Include the boolean key "is_valid_wine_query" in the JSON object.
3. Set "is_valid_wine_query": false if the prompt is meta-conversational, system/status checks (e.g., "is this working", "hello", "test", "who are you"), non-wine queries, or completely lacks wine attributes. Set "is_valid_wine_query": true ONLY if the user explicitly describes wine parameters or asks for a wine quality prediction based on wine traits.
"""


def get_client() -> OpenAI:
    """Initializes OpenAI client targeting Hugging Face Serverless Router Gateway."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable missing.")
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )


def extract_features_from_prompt(user_prompt: str, client: OpenAI = None) -> dict:
    """Parses natural language prompt into structured feature JSON."""
    if client is None:
        client = get_client()

    try:
        response = client.chat.completions.create(
            model=HF_MODEL_NAME,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return {}

    raw_text = response.choices[0].message.content.strip()
    cleaned_text = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"^```\s*", "", cleaned_text, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {"is_valid_wine_query": False, "error": "JSON parse error"}


def generate_conversational_response(
    user_prompt: str, quality_score: float, features: dict, missing_imputed: list, client: OpenAI = None
) -> str:
    """Generates natural language explanation summarizing model findings."""
    if client is None:
        client = get_client()

    system_prompt = """
You are an expert sommelier and data scientist.
Explain a wine quality prediction score concisely to a user in a helpful, conversational tone.

Guidelines:
1. Explain what a quality rating (scored 1 to 10) means for this specific profile.
2. Highlight key features (like alcohol, volatile acidity, or pH) driving the rating.
3. If any missing features were automatically filled with standard defaults, mention those caveats politely.
"""

    user_content = f"""
User Query: "{user_prompt}"
Model Predicted Quality Score: {quality_score:.2f} / 10.0
Used Feature Profile: {json.dumps(features)}
Features Automatically Imputed (Defaults Used): {missing_imputed}
"""

    response = client.chat.completions.create(
        model=HF_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()
