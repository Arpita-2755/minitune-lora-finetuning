import streamlit as st
import requests

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="MiniTune",
    page_icon="🧪"
)

st.title("🧪 MiniTune")
st.subheader("LoRA Fine-Tuned ML Viva Model (HF Hosted)")

# ---------------------------
# Secrets Check
# ---------------------------
if "HF_API_TOKEN" not in st.secrets:
    st.error("HF_API_TOKEN not found in Streamlit Secrets.")
    st.stop()

HF_API_TOKEN = st.secrets["HF_API_TOKEN"]

# ---------------------------
# Model Config
# ---------------------------
MODEL_ID = "arpitamishra27/minitune-merged-model"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}


# ---------------------------
# Query Function
# ---------------------------
def query_model(question):
    payload = {
        "inputs": f"Question: {question}\nAnswer:",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )
    except Exception as e:
        return f"Request failed: {str(e)}"

    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}"

    try:
        result = response.json()
    except Exception:
        return f"Invalid JSON response:\n{response.text}"

    if isinstance(result, list):
        return result[0].get("generated_text", result)

    return result


# ---------------------------
# UI Input
# ---------------------------
question = st.text_input("Enter an ML viva question")

if question:
    with st.spinner("Generating response..."):
        answer = query_model(question)

    st.markdown("### 🧠 Fine-Tuned Model Response")
    st.write(answer)
