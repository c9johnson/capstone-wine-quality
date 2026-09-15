import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import streamlit as st
from dotenv import load_dotenv
from src.llm_interface import extract_features_from_prompt, generate_conversational_response, get_client
from src.predict import load_latest_model, predict_wine_quality

load_dotenv()

st.set_page_config(page_title="Wine Quality Predictor & Sommelier", page_icon="🍷", layout="wide")

st.title("🍷 LLM-Powered Wine Quality Assistant")
st.markdown("Describe a wine in plain English and our Random Forest pipeline will predict its quality score!")

# Check API Token
if not os.getenv("HF_TOKEN"):
    st.error("⚠️ `HF_TOKEN` environment variable is not set. Please add `HF_TOKEN=your_token` to your `.env` file.")
    st.stop()

@st.cache_resource
def init_resources():
    client = get_client()
    model, config, best_run_id = load_latest_model()
    return client, model, config, best_run_id

try:
    client, model, config, best_run_id = init_resources()
    st.sidebar.header("📊 Active ML Model")
    st.sidebar.success(f"Champion Run ID: `{best_run_id[:8]}...`")
    st.sidebar.info(f"Model Type: `{config['model']['type']}`")
except Exception as e:
    st.sidebar.error(f"Initialization Error: {e}")
    st.stop()

# Initialize session state for prompt input text
if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = ""

# Sample Prompts
st.markdown("**Try a sample prompt:**")
col1, col2, col3 = st.columns(3)

if col1.button("🥂 Dry High-Alcohol White"):
    st.session_state["prompt_input"] = "A white wine with 12.8% alcohol, low residual sugar of 1.5, low volatile acidity, and a crisp pH of 3.1."
if col2.button("🍷 Full-Bodied Red"):
    st.session_state["prompt_input"] = "A rich red wine with 13.5% alcohol, high citric acid, volatile acidity of 0.45, and pH of 3.5."
if col3.button("❓ Out-of-Scope Query"):
    st.session_state["prompt_input"] = "What is the capital city of France?"

# Text area bound to session state via key="prompt_input"
user_input = st.text_area(
    "Enter wine description:",
    key="prompt_input",
    height=100,
    placeholder="e.g., Predict quality for a white wine with 11.5% alcohol, 3.2 pH, and low volatile acidity...",
)

if st.button("🔮 Predict Quality", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a description or pick a sample prompt above.")
    else:
        with st.spinner("Analyzing text with Hugging Face LLM & running ML Inference..."):
            extracted = extract_features_from_prompt(user_input, client)

            # Strict out-of-scope check (defaults to False if key is missing/unclear)
            if not extracted.get("is_valid_wine_query", False):
                st.error("⚠️ **Out-of-Scope Query Detected**")
                st.info("I am a wine quality prediction assistant. Please enter a query describing wine characteristics.")
            else:
                score, final_features, missing_imputed = predict_wine_quality(extracted, model, config)
                explanation = generate_conversational_response(user_input, score, final_features, missing_imputed, client)

                st.divider()
                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    st.metric("Predicted Quality Rating", f"{score:.2f} / 10.0")
                    st.progress(min(max(score / 10.0, 0.0), 1.0))

                    with st.expander("🔍 Extracted Feature JSON"):
                        st.json(final_features)
                        if missing_imputed:
                            st.caption(f"Defaults used for: {', '.join(missing_imputed)}")

                with res_col2:
                    st.subheader("Sommelier Analysis")
                    st.write(explanation)