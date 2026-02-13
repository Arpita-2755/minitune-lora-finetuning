import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="MiniTune", page_icon="🧪")

st.title("🧪 MiniTune")
st.subheader("Base vs LoRA Fine-Tuned Comparison")

BASE_MODEL = "google/flan-t5-small"
FINETUNED_MODEL = "arpitamishra27/minitune-merged-model"

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL)

    return tokenizer, base_model, finetuned_model

tokenizer, base_model, finetuned_model = load_models()


def generate_answer(model, question):
    prompt = f"""
You are an ML viva examiner assistant.
Answer the following machine learning question clearly and concisely.

Question: {question}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            min_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
