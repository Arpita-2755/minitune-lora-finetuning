import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="MiniTune", page_icon="🧪")

st.title("🧪 MiniTune")
st.subheader("LoRA Fine-Tuned ML Viva Model (Merged & Hosted)")

MODEL_PATH = "arpitamishra27/minitune-merged-model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

question = st.text_input("Enter an ML viva question")

if question:
    inputs = tokenizer(
        f"Question: {question}\nAnswer:",
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.markdown("### 🧠 Fine-Tuned Model Response")
    st.write(answer)
