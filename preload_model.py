"""
Run this once before the hackathon to pre-download the model into model_cache/.
This means the container starts instantly without downloading several GB.

Usage:
    HF_HOME=./model_cache python preload_model.py
"""
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch

MODEL_ID = "google/gemma-3n-E4B-it"

print(f"Downloading {MODEL_ID}...")
AutoProcessor.from_pretrained(MODEL_ID)
Gemma3nForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
print("Done. Model cached.")
