import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("MiniTune")
st.subheader("LoRA Fine-Tuned ML Viva Model")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("minitune-merged-model")
    model = AutoModelForSeq2SeqLM.from_pretrained("minitune-merged-model")
    return tokenizer, model

tokenizer, model = load_model()

def generate(question):
    prompt = f"You are a machine learning viva examiner assistant.\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=80,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = st.text_input("Enter ML viva question")

if question:
    st.write(generate(question))
