# inference.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATH = "minitune-lora-model"


def ask(question: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    top_p=0.9
)


    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(ask("What is overfitting in machine learning?"))
