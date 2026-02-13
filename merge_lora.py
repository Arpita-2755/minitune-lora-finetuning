from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "google/flan-t5-small"
ADAPTER_PATH = "minitune-lora-model"

print("Loading base model...")
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("minitune-merged-model")
tokenizer.save_pretrained("minitune-merged-model")

print("Done.")
