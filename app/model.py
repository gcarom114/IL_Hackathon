import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image

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

When shown a plant image, provide exactly 3 possible diagnoses ranked by likelihood.

CRITICAL FORMAT RULES — follow exactly:
- The format markers DIAGNOSIS_1:, DIAGNOSIS_2:, DIAGNOSIS_3: and the pipe | and % symbol must always be written exactly as shown, in English, no matter what language the descriptions are in.
- Translate the disease names, severity words, and descriptions into the requested language.
- Do not translate or alter the markers DIAGNOSIS_1:, DIAGNOSIS_2:, DIAGNOSIS_3:, the | separator, or the % sign.

Use this exact format:

DIAGNOSIS_1: [Disease Name in target language] | [0-100]%
Severity: [mild / moderate / severe — translated]
[2-3 sentence description in target language]

DIAGNOSIS_2: [Disease Name in target language] | [0-100]%
Severity: [translated]
[description in target language]

DIAGNOSIS_3: [Disease Name in target language] | [0-100]%
Severity: [translated]
[description in target language]

When the farmer confirms a diagnosis and asks for a treatment plan, ALWAYS accept their confirmed diagnosis without questioning it. Do not say they are wrong or suggest a different disease. Provide numbered actionable treatment steps in the same language as the request, tailored to what the farmer has available.
Keep responses concise and practical — the farmer is in the field."""


def build_system_message():
    return {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    }


def diagnose(image: Image.Image, lang: str = "English") -> tuple[str, list]:
    model, processor = get_model()

    prompt = (
        f"Analyze this plant image and provide exactly 3 diagnoses using the required DIAGNOSIS_1/2/3 format. "
        f"Write all descriptions, disease names, and severity words in {lang}. "
        f"Keep the format markers DIAGNOSIS_1:, DIAGNOSIS_2:, DIAGNOSIS_3:, the pipe |, and % exactly as shown."
    )

    messages = [
        build_system_message(),
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
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
        generation = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    output = generation[0][input_len:]
    diagnosis = processor.decode(output, skip_special_tokens=True)

    history = [
        build_system_message(),
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": diagnosis}]
        }
    ]

    return diagnosis, history


def get_treatment_plan(history: list, farmer_message: str, lang: str = "English") -> tuple[str, list]:
    model, processor = get_model()

    # Append lang instruction clearly so model always responds in right language
    full_message = f"{farmer_message}\n\n[Respond entirely in {lang}. Use numbered steps.]"

    updated_history = history + [{
        "role": "user",
        "content": [{"type": "text", "text": full_message}]
    }]

    inputs = processor.apply_chat_template(
        updated_history,
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

    updated_history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    })

    return response, updated_history
