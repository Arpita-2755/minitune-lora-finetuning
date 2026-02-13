import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="MiniTune", page_icon="🧪")

st.title("🧪 MiniTune")
st.subheader("Base vs LoRA Fine-Tuned Comparison")

BASE_MODEL = "google/flan-t5-small"
FINETUNED_MODEL = "arpitamishra27/minitune-merged-model"

# ---------------------------
# Load Models (Cached)
# ---------------------------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL)

    return tokenizer, base_model, finetuned_model


tokenizer, base_model, finetuned_model = load_models()

# ---------------------------
# Generation Function
# ---------------------------
def generate_answer(model, question):
    # IMPORTANT: Match training format exactly
    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.3
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------
# UI
# ---------------------------
question = st.text_input("Enter an ML viva question")

if question:
    with st.spinner("Generating responses..."):
        base_answer = generate_answer(base_model, question)
        finetuned_answer = generate_answer(finetuned_model, question)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Base Model")
        st.write(base_answer)

    with col2:
        st.markdown("### 🧪 Fine-Tuned Model")
        st.write(finetuned_answer)
