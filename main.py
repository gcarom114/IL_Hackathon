"""
Plant issue triage CLI for the InstaLily hackathon.

Workflow covered:
1) User snaps a plant photo and runs `python main.py analyze <image>`.
2) App returns the top three likely issues with confidence scores.
3) User asks for next steps with `python main.py next-steps "<issue name>"`.

The Gemma3nAgent class is a thin shim where a real Gemma 3n endpoint can be
plugged in. For now it uses light-weight rule-based reasoning so the script
works offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import typer
from PIL import Image, UnidentifiedImageError
import requests

app = typer.Typer(add_completion=False, help="Plant diagnostic helper using a Gemma 3n-style agent.")


# -------------------- Core domain models -------------------- #
@dataclass
class IssueHypothesis:
    name: str
    confidence: float
    rationale: str


@dataclass
class AnalysisResult:
    image: Path
    notes: str
    hypotheses: List[IssueHypothesis]
    telemetry: Dict[str, float]


# -------------------- Image feature extraction -------------------- #
def load_image(path: Path) -> np.ndarray:
    """Load image as float32 RGB array in range [0, 1]."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def extract_features(image: np.ndarray) -> Dict[str, float]:
    """
    Lightweight heuristic features:
    - green_level: healthy foliage baseline
    - yellowing: chlorosis indicator
    - brown_fraction: dryness/burn indicator
    - dark_spot_fraction: fungal/rot indicator
    - edge_contrast: mechanical damage or pest chewing
    """
    green_level = float(np.mean(image[:, :, 1]))
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    yellowing = float(np.mean((red + green) / 2 - blue))
    brown_mask = (red > 0.35) & (green > 0.2) & (green < red * 0.9)
    brown_fraction = float(np.mean(brown_mask))

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    dark_spot_fraction = float(np.mean(luminance < 0.25))

    # Edge contrast via simple horizontal/vertical diff
    h_diff = np.mean(np.abs(image[1:, :, :] - image[:-1, :, :]))
    v_diff = np.mean(np.abs(image[:, 1:, :] - image[:, :-1, :]))
    edge_contrast = float((h_diff + v_diff) / 2)

    return {
        "green_level": green_level,
        "yellowing": yellowing,
        "brown_fraction": brown_fraction,
        "dark_spot_fraction": dark_spot_fraction,
        "edge_contrast": edge_contrast,
    }


# -------------------- Gemma 3n agent shim -------------------- #
class Gemma3nAgent:
    """
    Abstraction layer where a real Gemma 3n model call can be wired in.
    Set GEMMA3N_ENDPOINT to enable your own HTTP endpoint and toggle
    `use_llm=True` when constructing the agent.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm and bool(os.getenv("GEMMA3N_ENDPOINT"))
        self.endpoint = os.getenv("GEMMA3N_ENDPOINT")

    def _llm_call(self, payload: Dict) -> Optional[List[IssueHypothesis]]:
        """
        POSTs the structured payload to the GEMMA3N_ENDPOINT.
        Expected JSON response shape (flexible):
        {
          "hypotheses": [
            {"name": "Early blight", "confidence": 0.62, "rationale": "…"},
            ...
          ]
        }
        If the endpoint returns plain text, we attempt to parse line-wise.
        Returns None on any failure to allow heuristic fallback.
        """
        if not self.endpoint:
            return None

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"[Gemma3nAgent] LLM call failed: {exc}; falling back to heuristics.")
            return None

        try:
            data = resp.json()
        except ValueError:
            text = resp.text
            return self._parse_text_response(text)

        if isinstance(data, dict) and "hypotheses" in data and isinstance(data["hypotheses"], list):
            parsed: List[IssueHypothesis] = []
            for item in data["hypotheses"][:3]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "Unknown issue"))
                conf = float(item.get("confidence", 0.0))
                rationale = str(item.get("rationale", ""))
                parsed.append(IssueHypothesis(name, conf, rationale))
            if parsed:
                return parsed

        # Fall back to parsing if the shape is unexpected
        return self._parse_text_response(resp.text)

    def _parse_text_response(self, text: str) -> Optional[List[IssueHypothesis]]:
        """
        Very lightweight parser for text responses where the model returns
        bullet lines like '1. Leaf spot (0.61) - dark lesions...'
        """
        lines = [ln.strip(" -") for ln in text.splitlines() if ln.strip()]
        hyps: List[IssueHypothesis] = []
        for ln in lines:
            # Try to extract leading rank and confidence in parentheses
            # Examples: "1. Early blight (0.64) - dark lesions"
            name = ln
            conf = 0.0
            if "(" in ln and ")" in ln:
                try:
                    before, after = ln.split("(", 1)
                    conf_str, rest = after.split(")", 1)
                    conf = float(conf_str.strip())
                    name = before.replace(".", " ").strip() + rest
                except Exception:
                    pass
            hyps.append(IssueHypothesis(name=name.strip(), confidence=conf, rationale=""))
            if len(hyps) == 3:
                break
        return hyps or None

    def rank_issues(self, feats: Dict[str, float], notes: str) -> List[IssueHypothesis]:
        # Optional LLM path: you can build a JSON spec and let Gemma respond.
        if self.use_llm:
            payload = {"features": feats, "notes": notes, "top_k": 3}
            llm_hyps = self._llm_call(payload)
            if llm_hyps:
                return llm_hyps

        candidates: List[Tuple[str, float, str]] = []
        chlorosis = max(0.0, feats["yellowing"] * 1.5)
        dryness = feats["brown_fraction"]
        fungal = feats["dark_spot_fraction"]
        chewing = feats["edge_contrast"]

        candidates.append(
            (
                "Nitrogen deficiency (chlorosis)",
                min(1.0, chlorosis),
                "Yellowing relative to green baseline suggests chlorosis.",
            )
        )
        candidates.append(
            (
                "Under-watering or heat scorch",
                min(1.0, dryness * 1.2),
                "Brown desiccated patches are consistent with water stress.",
            )
        )
        candidates.append(
            (
                "Fungal leaf spot / early blight",
                min(1.0, fungal * 1.4),
                "Dark low-luminance spots indicate possible fungal lesions.",
            )
        )
        candidates.append(
            (
                "Chewing pest damage",
                min(1.0, chewing * 1.1),
                "High edge contrast may come from irregular bite marks.",
            )
        )

        # Lightly boost hypotheses that match user-provided notes keywords
        notes_lower = notes.lower()
        for idx, (name, score, rationale) in enumerate(candidates):
            if any(k in notes_lower for k in ["bug", "chew", "hole", "aphid"]):
                if "pest" in name:
                    score += 0.1
            if "yellow" in notes_lower and "chlorosis" in name:
                score += 0.05
            candidates[idx] = (name, min(score, 1.0), rationale)

        ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:3]
        return [IssueHypothesis(n, round(c, 3), r) for n, c, r in ranked]

    def suggest_next_steps(self, issue: str) -> List[str]:
        issue_lower = issue.lower()
        if "chlorosis" in issue_lower or "nitrogen" in issue_lower:
            return [
                "Apply a balanced fertilizer with higher nitrogen (e.g., 10-5-5).",
                "Water thoroughly after feeding to avoid root burn.",
                "Monitor new growth over the next 7–10 days for greening.",
            ]
        if "fungal" in issue_lower or "blight" in issue_lower or "spot" in issue_lower:
            return [
                "Prune and discard affected leaves; do not compost.",
                "Apply a copper-based fungicide or chlorothalonil per label.",
                "Improve airflow: reduce overhead watering and increase spacing.",
            ]
        if "pest" in issue_lower or "chew" in issue_lower:
            return [
                "Inspect leaf undersides for aphids, caterpillars, or beetles.",
                "Use neem oil or spinosad; repeat in 7 days if activity continues.",
                "For heavy infestations, consider a targeted insecticidal soap.",
            ]
        if "water" in issue_lower or "scorch" in issue_lower:
            return [
                "Deep-water the root zone and add mulch to retain moisture.",
                "Provide afternoon shade if temperatures exceed 90°F (32°C).",
                "Check soil moisture daily for a week; avoid keeping soil soggy.",
            ]
        return [
            "Gather more context: soil moisture, recent weather, and fertilizer use.",
            "Isolate the plant if symptoms are spreading.",
            "Consult a local extension office with clear photos for confirmation.",
        ]


# -------------------- CLI commands -------------------- #
@app.command()
def analyze(
    image_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the plant photo"),
    notes: str = typer.Option("", "--notes", "-n", help="Optional observations (e.g., smell, time of day, pests seen)"),
):
    """Analyze a plant photo and return the top three likely issues."""
    try:
        image = load_image(image_path)
    except UnidentifiedImageError:
        typer.echo("Error: Unsupported image format. Please provide a common format like JPG or PNG.")
        raise typer.Exit(code=1)

    features = extract_features(image)
    agent = Gemma3nAgent(use_llm=bool(os.getenv("GEMMA3N_ENDPOINT")))
    hypotheses = agent.rank_issues(features, notes)

    result = AnalysisResult(
        image=image_path,
        notes=notes,
        hypotheses=hypotheses,
        telemetry=features,
    )

    typer.echo(f"Image: {image_path}")
    if notes:
        typer.echo(f"Notes: {notes}")
    typer.echo("\nTop 3 suspected issues:")
    for idx, h in enumerate(result.hypotheses, start=1):
        typer.echo(f"{idx}. {h.name} | confidence={h.confidence:.2f} | {h.rationale}")

    typer.echo("\nTelemetry (for debugging / Gemma prompt enrichment):")
    for k, v in result.telemetry.items():
        typer.echo(f"- {k}: {v:.3f}")


@app.command("next-steps")
def next_steps(
    issue: str = typer.Argument(..., help="Issue label from the analyze step"),
):
    """Provide actionable remediation guidance for a selected issue."""
    agent = Gemma3nAgent(use_llm=bool(os.getenv("GEMMA3N_ENDPOINT")))
    steps = agent.suggest_next_steps(issue)

    typer.echo(f"Recommended actions for: {issue}")
    for idx, step in enumerate(steps, start=1):
        typer.echo(f"{idx}. {step}")


if __name__ == "__main__":
    app()
