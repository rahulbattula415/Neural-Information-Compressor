# scripts/freeze_model.py
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    revision="607a30d",  # pin this — find via HuggingFace commit history
)
model.save_pretrained("nic/model/")