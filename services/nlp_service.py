"""NLP model service: load, predict top-k motive probabilities from narrative text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config import MODELS, PATHS
from services.tabular_service import ModelNotAvailableError


def _nlp_candidate_dirs() -> List[Path]:
    candidates: List[Path] = []
    for candidate in (PATHS.nlp_model_dir, PATHS.nlp_model_dir.parent):
        path = Path(candidate)
        if path not in candidates:
            candidates.append(path)
    return candidates


def _has_model_artifacts(base_dir: Path) -> bool:
    if not base_dir.exists() or not base_dir.is_dir():
        return False
    has_config = (base_dir / "config.json").exists()
    has_weights = any(
        (base_dir / filename).exists()
        for filename in ("model.bin", "pytorch_model.bin", "model.safetensors")
    )
    return has_config and has_weights


def _load_label_map(base_dir: Path) -> Dict[int, str]:
    label_map_path = base_dir / "label_map.json"
    if not label_map_path.exists():
        return {}
    try:
        raw = json.loads(label_map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    normalized: Dict[int, str] = {}
    if not isinstance(raw, dict):
        return normalized

    for key, value in raw.items():
        try:
            normalized[int(key)] = str(value)
        except Exception:
            continue
    return normalized


def _friendly_error_nlp(base_dir: Path | None = None) -> str:
    """Build a friendly error message for missing NLP artifacts."""
    searched = "\n".join(f"  - {path}" for path in _nlp_candidate_dirs())
    expected = base_dir or PATHS.nlp_model_dir
    return (
        "NLP model files were not found.\n\n"
        f"Expected at: {expected}\n\n"
        "Searched:\n"
        f"{searched}\n\n"
        "To enable text-based motive predictions, place your trained NLP model in:\n"
        f"  - {PATHS.nlp_model_dir}\n"
        f"  - {PATHS.nlp_model_dir.parent}\n\n"
        "Update `services.nlp_service.load_nlp_model` if your model uses a different structure."
    )


@st.cache_resource(show_spinner=False)
def load_nlp_model() -> Dict[str, Any]:
    """Load the NLP model artefacts.

    Expects a local transformer model directory under ``artifacts/nlp/model/``
    or ``artifacts/nlp/`` with at least ``config.json`` and weights.
    """
    for model_dir in _nlp_candidate_dirs():
        if _has_model_artifacts(model_dir):
            return {"model_path": str(model_dir), "version": MODELS.nlp_model_version}

    raise ModelNotAvailableError(_friendly_error_nlp())


def predict_nlp_topk(text: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Run NLP model to get motive probabilities.

    Returns a dictionary::

        {
            "topk": [{"label": ..., "prob": ...}, ...],
            "pred": "<top1_label>",
            "confidence": <top1_prob>
        }

    or ``None`` if the input text is empty. Only a real, successfully loaded
    model may produce top-k probabilities. If no model is available, the
    function returns an unavailable status with an empty top-k list.
    """
    if not text or not str(text).strip():
        return None

    def format_unavailable(reason: str, error: Exception | None = None) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "topk": [],
            "pred": "",
            "confidence": 0.0,
            "source": "unavailable",
            "reason": reason,
        }
        if error is not None:
            output["error"] = str(error)
        return output

    def format_output(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = [
            {
                "label": str(item.get("label", "")),
                "prob": float(item.get("prob", 0.0) or 0.0),
            }
            for item in lst
        ]
        if not normalized:
            return {"topk": [], "pred": "", "confidence": 0.0, "source": "real"}
        pred = normalized[0].get("label", "")
        conf = float(normalized[0].get("prob", 0.0) or 0.0)
        return {"topk": normalized, "pred": pred, "confidence": conf, "source": "real"}

    try:
        # ensure model artefacts exist; may raise ModelNotAvailableError
        model_info = load_nlp_model()

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
            model_dir = Path(path)
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForSequenceClassification.from_pretrained(path)
            label_map = _load_label_map(model_dir)
            if label_map:
                mdl.config.id2label = dict(label_map)
                mdl.config.label2id = {label: idx for idx, label in label_map.items()}
            return tok, mdl, label_map

        model_path = str(model_info["model_path"])
        tokenizer, model, label_map = _cached_transformer(model_path)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
        # softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        labels = [label_map.get(i) or model.config.id2label.get(i, str(i)) for i in range(len(probs))]
        idx_sorted = probs.argsort()[::-1][:top_k]
        topk_list = [{"label": labels[i], "prob": float(probs[i])} for i in idx_sorted]
        return format_output(topk_list)

    except ModelNotAvailableError:
        return format_unavailable("model_not_available")
    except Exception as e:
        # any other failure (import issue, load error, inference error, etc.)
        return format_unavailable("load_or_inference_error", error=e)


__all__ = ["load_nlp_model", "predict_nlp_topk"]
