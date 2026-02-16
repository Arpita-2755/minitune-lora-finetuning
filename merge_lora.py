# merge_lora.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "google/flan-t5-base"
LORA_MODEL = "minitune-lora-model"
MERGED_MODEL = "minitune-merged-model"

print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

print("Loading LoRA model...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(MERGED_MODEL)

# 🔹 Save tokenizer properly
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_MODEL)

print("Merged model saved successfully.")
