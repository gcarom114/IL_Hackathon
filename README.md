# InstaLily Plant Triage (Gemma 3n Friendly)

CLI workflow:

1) Capture a photo of the affected plant.
2) Run `python main.py analyze <image> --notes "your observations"`.
3) Review the top three likely issues and confidence scores.
4) Run `python main.py next-steps "<issue name>"` to get remediation guidance.

Setup:
- python 3.10+ recommended
- `pip install -r requirements.txt`

Gemma 3n integration:
- Set `GEMMA3N_ENDPOINT` to a callable endpoint that accepts your prompt/payload.
- Construct `Gemma3nAgent(use_llm=True)` if you patch in the real call inside `_llm_call`.
- The agent currently falls back to lightweight heuristics so it works offline.

Notes:
- The heuristics use simple color and contrast statistics; they are transparent but not a substitute for a trained vision model.
- The CLI echoes a small telemetry block you can feed into your Gemma prompt for richer reasoning.
