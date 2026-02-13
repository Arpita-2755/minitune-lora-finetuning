# train_lora.py

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "google/flan-t5-small"  # CPU-friendly & instruction-tuned


def load_dataset(path="data/ml_viva_qa.json"):
    with open(path, "r") as f:
        data = json.load(f)

    inputs = [
        f"Question: {item['question']}\nAnswer:"
        for item in data
    ]
    targets = [item["answer"] for item in data]

    return Dataset.from_dict({"input_text": inputs, "target_text": targets})


def tokenize(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    labels = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # ---- LoRA configuration ----
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset()
    tokenized_ds = dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True
    )

    training_args = TrainingArguments(
        output_dir="./minitune-output",
        per_device_train_batch_size=2,
        num_train_epochs=10,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds
    )

    trainer.train()

    model.save_pretrained("minitune-lora-model")
    tokenizer.save_pretrained("minitune-lora-model")


if __name__ == "__main__":
    main()
