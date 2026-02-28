"""
Mock model for testing the full API flow without loading Gemma 3n.
Swap this in api.py during development, swap back when model is ready.
"""

from PIL import Image


def diagnose(image: Image.Image) -> tuple[str, list]:
    diagnosis = (
        "Early Blight (Alternaria solani) — 82% confidence. "
        "Dark concentric lesions are visible on the lower leaves, "
        "characteristic of fungal infection. Spreads rapidly in warm, "
        "humid conditions. Immediate treatment is recommended before "
        "the infection reaches upper foliage."
    )

    history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert agronomist."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What disease do you see?"}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": diagnosis}]
        }
    ]

    return diagnosis, history


def get_treatment_plan(history: list, farmer_message: str) -> tuple[str, list]:
    response = (
        "Based on your situation, here is your treatment plan:\n\n"
        "1. Apply copper fungicide at 2 tbsp per gallon of water across all affected areas. "
        "For 2 acres, you will need approximately 4–5 gallons of solution.\n\n"
        "2. Spray early morning so leaves dry before evening — wet foliage overnight "
        "accelerates fungal spread. Since it rained this morning, wait until tomorrow.\n\n"
        "3. Remove and bag the most severely affected lower leaves today. "
        "Do not compost them — dispose of them away from the field.\n\n"
        "4. Reapply fungicide in 7 days. Monitor new growth closely for the next two weeks."
    )

    history.append({
        "role": "user",
        "content": [{"type": "text", "text": farmer_message}]
    })
    history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    })

    return response, history
