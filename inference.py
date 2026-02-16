# inference.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = "minitune-merged-model"

print("Loading merged model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_answer(question):
    prompt = (
        "Answer the following machine learning viva question clearly and concisely.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt part if repeated
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer.strip()


while True:
    question = input("\nEnter ML Question (or 'exit'): ")
    if question.lower() == "exit":
        break

    response = generate_answer(question)
    print("\n🧠 Model Answer:")
    print(response)
