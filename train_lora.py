"""
Lightweight QLoRA fine-tuning script for Gemma 3n on plant diagnosis data.

Dataset format (JSONL):
{"image_path": "data/img1.jpg", "prompt": "What disease do you see?", "response": "Early blight (~0.82). Dark concentric lesions."}

Run:
HF_TOKEN=... python train_lora.py --data data/train.jsonl --epochs 1 --batch 1 --grad-accum 8 --output lora-out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from unsloth import FastVisionModel


SYSTEM_PROMPT = (
    "You are an expert agronomist and plant pathologist with decades of field experience. "
    "When shown a plant image, identify any visible diseases, pests, or deficiencies. "
    "Be specific: state the disease name, an estimated confidence level, and a brief explanation. "
    "Keep responses concise and practical — the farmer is in the field."
)


def build_system_message():
    return {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}


class PlantDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor, image_size: int = 512):
        self.items: List[Dict] = []
        self.processor = processor
        self.image_size = image_size
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.items[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image.thumbnail((self.image_size, self.image_size))

        # Build messages without the assistant turn to measure prompt length
        messages = [
            build_system_message(),
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
        ]

        prompt_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Add the ground-truth assistant reply
        full_messages = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": sample["response"]}]}
        ]
        model_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = input_ids.clone()

        prompt_len = prompt_inputs["input_ids"].shape[-1]
        labels[:, :prompt_len] = -100  # ignore prompt tokens

        model_inputs["labels"] = labels
        return model_inputs


def collate_batch(features: List[Dict], pad_token_id: int):
    input_ids = [f["input_ids"][0] for f in features]
    attention_masks = [f["attention_mask"][0] for f in features]
    labels = [f["labels"][0] for f in features]
    pixel_values = [f["pixel_values"][0] for f in features]

    def pad_stack(tensors, pad_value):
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                t = torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)

    batch = {
        "input_ids": pad_stack(input_ids, pad_token_id),
        "attention_mask": pad_stack(attention_masks, 0),
        "labels": pad_stack(labels, -100),
        "pixel_values": torch.stack(pixel_values, dim=0),
    }
    return batch


def parse_args():
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Gemma 3n for plant diagnosis.")
    ap.add_argument("--data", required=True, help="Path to training JSONL.")
    ap.add_argument("--model", default="unsloth/gemma-3n-e4b-it", help="Base model id.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1, help="Per-device batch size.")
    ap.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output", default="lora-out", help="Where to save the adapter.")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")

    print(f"Loading base model {args.model} (4-bit) ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=True,
        token=hf_token,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
    )

    processor = tokenizer

    dataset = PlantDataset(Path(args.data), processor, image_size=args.image_size)
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda feats: collate_batch(feats, pad_token_id),
    )

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if args.max_steps and step >= args.max_steps:
                break

        if args.max_steps and step >= args.max_steps:
            break

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
