import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import os

MODEL_ID = "google/gemma-3n-E4B-it"

model = None
processor = None


def load_model():
    global model, processor
    print(f"Loading {MODEL_ID}...")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.")


def get_model():
    if model is None or processor is None:
        load_model()
    return model, processor


SYSTEM_PROMPT = """You are an expert agronomist and plant pathologist with decades of field experience.
When shown a plant image, identify any visible diseases, pests, or deficiencies.
Be specific: state the disease name, an estimated confidence level, and a brief explanation.
When asked for a treatment plan, give actionable advice tailored to what the farmer has available.
Keep responses concise and practical — the farmer is in the field."""


def build_system_message():
    return {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    }


def diagnose(image: Image.Image) -> tuple[str, list]:
    """
    Takes a PIL image, returns (diagnosis_text, conversation_history).
    conversation_history is passed back into get_treatment_plan.
    """
    model, processor = get_model()

    messages = [
        build_system_message(),
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What disease, pest, or deficiency do you see on this plant? Give your diagnosis and confidence level."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    output = generation[0][input_len:]
    diagnosis = processor.decode(output, skip_special_tokens=True)

    # Build history for follow-up turns
    history = [
        build_system_message(),
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What disease, pest, or deficiency do you see on this plant?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": diagnosis}]
        }
    ]

    return diagnosis, history


def get_treatment_plan(history: list, farmer_message: str) -> tuple[str, list]:
    """
    Takes existing conversation history and farmer's follow-up message.
    Returns (treatment_plan_text, updated_history).
    """
    model, processor = get_model()

    history.append({
        "role": "user",
        "content": [{"type": "text", "text": farmer_message}]
    })

    inputs = processor.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=400, do_sample=False)

    output = generation[0][input_len:]
    response = processor.decode(output, skip_special_tokens=True)

    history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    })

    return response, history
