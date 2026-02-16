# train_lora.py

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "google/flan-t5-base"
OUTPUT_DIR = "minitune-lora-model"

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

print("Loading dataset...")
with open("data/ml_viva_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# 🔹 Instruction formatting
def format_example(example):
    prompt = (
        "Answer the following machine learning viva question clearly and concisely.\n\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )
    return {
        "input_text": prompt,
        "target_text": example["answer"]
    }

dataset = dataset.map(format_example)

# 🔹 Tokenization
def tokenize_function(example):
    model_inputs = tokenizer(
        example["input_text"],
        max_length=256,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        example["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

# 🔹 LoRA config
print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)

# 🔹 Data collator (important fix)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("Training...")
trainer.train()

print("Saving LoRA model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete.")
