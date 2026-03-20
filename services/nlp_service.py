"""NLP model service: load, predict top-k motive probabilities from narrative text."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config import MODELS, PATHS
from services.tabular_service import ModelNotAvailableError


def _friendly_error_nlp(base_dir: Path) -> str:
    """Build a friendly error message for missing NLP artifacts."""
    return (
        "NLP model files were not found.\n\n"
        f"Expected at: {base_dir}\n\n"
        "To enable text-based motive predictions, place your trained NLP model in:\n"
        f"  - {PATHS.nlp_model_dir}\n\n"
        "Update `services.nlp_service.load_nlp_model` if your model uses a different structure."
    )


@st.cache_resource(show_spinner=False)
def load_nlp_model() -> Dict[str, Any]:
    """Load the NLP model artefacts.

    Expects artifacts/nlp/model/ to contain the model files.
    """
    model_dir: Path = PATHS.nlp_model_dir
    # Support both model.bin (legacy) and standard transformer/model dirs
    model_file = model_dir / "model.bin"
    if not model_file.exists():
        # Check for pytorch_model.bin, model.safetensors, or config.json (transformer)
        if not model_dir.exists():
            raise ModelNotAvailableError(_friendly_error_nlp(model_dir))
        # If dir exists but no model.bin, check for config.json (HuggingFace style)
        config_file = model_dir / "config.json"
        if not config_file.exists():
            raise ModelNotAvailableError(_friendly_error_nlp(model_dir))

    return {"model_path": str(model_dir), "version": MODELS.nlp_model_version}


def predict_nlp_topk(text: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Run NLP model to get motive probabilities.

    Returns a dictionary::

        {
            "topk": [{"label": ..., "prob": ...}, ...],
            "pred": "<top1_label>",
            "confidence": <top1_prob>
        }

    or ``None`` if the input text is empty.  If the real transformer model
    is available the function will use it; otherwise a mock or default
    response is returned.  Probabilities are normalised to sum to 1.0.
    """
    if not text or not str(text).strip():
        return None

    def format_output(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = [
            {
                "label": str(item.get("label", "")),
                "prob": float(item.get("prob", 0.0) or 0.0),
            }
            for item in lst
        ]
        if not normalized:
            return {"topk": [], "pred": "", "confidence": 0.0}
        pred = normalized[0].get("label", "")
        conf = float(normalized[0].get("prob", 0.0) or 0.0)
        return {"topk": normalized, "pred": pred, "confidence": conf}

    try:
        # ensure model artefacts exist; may raise ModelNotAvailableError
        load_nlp_model()

        # attempt to perform real transformer inference
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            import numpy as np
        except ImportError:
            # missing libraries; fall back to simple response below
            raise RuntimeError("transformers/torch not installed")

        # cache the tokenizer and model to avoid reloads on every call
        @st.cache_resource(show_spinner=False)
        def _cached_transformer(path: str):
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForSequenceClassification.from_pretrained(path)
            return tok, mdl

        model_path = PATHS.nlp_model_dir if isinstance(PATHS.nlp_model_dir, str) else str(PATHS.nlp_model_dir)
        tokenizer, model = _cached_transformer(model_path)

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
        # softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        labels = [model.config.id2label.get(i, str(i)) for i in range(len(probs))]
        idx_sorted = probs.argsort()[::-1][:top_k]
        topk_list = [{"label": labels[i], "prob": float(probs[i])} for i in idx_sorted]
        return format_output(topk_list)

    except ModelNotAvailableError:
        # no model artefacts, produce a heuristic mock list
        lowered = text.lower()
        candidates = [
            ("Control / dominance (mock)", 0.4 if "stalking" in lowered or "control" in lowered else 0.1),
            ("Property-focused (mock)", 0.35 if "burglary" in lowered or "theft" in lowered else 0.15),
            ("Opportunistic (mock)", 0.25 if "random" in lowered or "chance" in lowered else 0.1),
            ("Revenge / grievance (mock)", 0.2 if "revenge" in lowered or "grudge" in lowered else 0.05),
            ("Financial gain (mock)", 0.15 if "money" in lowered or "fraud" in lowered else 0.05),
        ]
        total = sum(p for _, p in candidates)
        if total <= 0:
            total = 1.0
        topk_list = [{"label": lbl, "prob": round(p / total, 4)} for lbl, p in candidates[:top_k]]
        return format_output(topk_list)
    except Exception as e:
        # any other failure (import issue, inference error, etc.)
        st.error(f"NLP inference error: {e}")
        return None


__all__ = ["load_nlp_model", "predict_nlp_topk"]
