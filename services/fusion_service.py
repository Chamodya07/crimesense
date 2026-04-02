"""Late fusion: combine tabular multi-target outputs with NLP motive top-k into one response."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from config import MODELS


def _nlp_topk_items(nlp_topk: Optional[Any]) -> List[Dict[str, Any]]:
    if not nlp_topk:
        return []
    if isinstance(nlp_topk, dict):
        topk = nlp_topk.get("topk", [])
        return topk if isinstance(topk, list) else []
    return nlp_topk if isinstance(nlp_topk, list) else []


def nlp_topk_to_probs(nlp_topk: Optional[Any]) -> Dict[str, float]:
    """Convert NLP output (list or dict) to normalized label->prob mapping.

    The prediction may be a raw list of ``{"label","prob"}`` pairs or the
    richer dictionary returned by :func:`predict_nlp_topk`. If the argument is
    ``None`` or contains no usable scores an empty dict is returned.
    """
    topk = _nlp_topk_items(nlp_topk)
    total = sum(float(x.get("prob", 0) or 0) for x in topk)
    if total <= 0:
        return {}
    return {str(x.get("label", "")): float(x.get("prob", 0) or 0) / total for x in topk}


def fuse_motive(
    tab_label: Optional[str],
    nlp_probs: Dict[str, float],
    w_tab: float = 0.35,
    w_nlp: float = 0.65,
) -> Dict[str, Any]:
    """Fuse tabular motive label with NLP probs; return motive pred/conf/band/probs.

    - Boost tabular motive if present, then renormalize.
    - Choose argmax for final motive.
    - Confidence band: High>=0.70, Medium>=0.50 else Low.
    """
    combined: Dict[str, float] = dict(nlp_probs)
    if tab_label and str(tab_label).strip() and tab_label != "Unknown":
        combined[tab_label] = combined.get(tab_label, 0) * w_nlp + w_tab
    if not combined:
        return {
            "pred": tab_label or "Unknown",
            "conf": 0.0,
            "band": "Low",
            "probs": {},
        }
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}
    pred = max(combined, key=combined.get)
    conf = combined[pred]
    band = "High" if conf >= 0.70 else "Medium" if conf >= 0.50 else "Low"
    return {
        "pred": pred,
        "conf": round(conf, 4),
        "band": band,
        "probs": combined,
    }


def late_fusion_predict(
    tab_pred: Dict[str, Any],
    shap_top: List[Tuple[str, float]],
    nlp_topk: Optional[Any],
    motive_key: str = "motive",
) -> Dict[str, Any]:
    """Late fusion: combine tabular + NLP into one unified response object.

    - All tabular targets are retained. The motive target is fused with NLP
      probabilities only when a real NLP output is present.
    - ``tab_pred`` may be either the old flattened dict or the newer structure
      containing a ``pred`` field; code handles both.
    """
    warnings: List[str] = []

    flat_tab = tab_pred.get("pred") if "pred" in tab_pred else tab_pred

    if motive_key not in flat_tab:
        for candidate in ("motive", "primary_motive", "crime_motive"):
            if candidate in flat_tab:
                motive_key = candidate
                break

    tab_motive = flat_tab.get(motive_key)
    nlp_topk_rows = _nlp_topk_items(nlp_topk)
    nlp_probs = nlp_topk_to_probs(nlp_topk)
    has_nlp_signal = bool(nlp_probs)
    nlp_source = nlp_topk.get("source", "") if isinstance(nlp_topk, dict) else ("real" if nlp_topk_rows else "")
    nlp_reason = nlp_topk.get("reason", "") if isinstance(nlp_topk, dict) else ""

    if not has_nlp_signal:
        if nlp_source == "unavailable":
            warnings.append("NLP motive model unavailable. Final motive shown from tabular output only.")
        else:
            warnings.append("Narrative missing or empty - NLP motive prediction skipped.")
        motive_result = {
            "pred": tab_motive if tab_motive else "Unknown",
            "conf": 0.0,
            "band": "Low",
            "probs": {},
        }
    else:
        motive_result = fuse_motive(tab_motive, nlp_probs, w_tab=0.35, w_nlp=0.65)
        if tab_motive and tab_motive != "Unknown":
            nlp_top1 = max(nlp_probs, key=nlp_probs.get) if nlp_probs else ""
            if nlp_top1 and nlp_top1 != tab_motive:
                warnings.append(
                    f"Tabular motive ({tab_motive}) differs from NLP top-1 ({nlp_top1}); fused result used."
                )
        if motive_result["conf"] < 0.50:
            warnings.append(f"Fused motive confidence ({motive_result['conf']:.2f}) is below 0.50.")

    final: Dict[str, Any] = {}
    for key, value in flat_tab.items():
        if key != motive_key:
            final[key] = value
    final[motive_key] = motive_result

    fusion_rationale = (
        "Motive fused from tabular label and NLP narrative top-k probabilities "
        "(w_tab=0.35, w_nlp=0.65)."
    )
    if not has_nlp_signal:
        if nlp_source == "unavailable":
            fusion_rationale = "Motive from tabular model only; NLP motive model is unavailable."
        else:
            fusion_rationale = "Motive from tabular model only; no narrative provided for NLP."

    tabular_summary = {
        "risk_category": flat_tab.get("risk_level") or flat_tab.get("risk_category", "-"),
        "risk_score": flat_tab.get("risk_score", "-"),
        "motive": flat_tab.get(motive_key, "-"),
        "experience_level": flat_tab.get("experience_level", "-"),
    }
    nlp_summary = {
        "risk_category": "-",
        "risk_score": "-",
        "motive": nlp_topk.get("pred") if isinstance(nlp_topk, dict) else (nlp_topk_rows[0]["label"] if nlp_topk_rows else "-"),
        "experience_level": "-",
        "key_phrases": [],
        "topk": nlp_topk_rows if has_nlp_signal else [],
        "source": nlp_source,
        "reason": nlp_reason,
    }

    return {
        "final": final,
        "fusion_meta": {
            "method": "late_fusion",
            "weights": {"w_tab": 0.35, "w_nlp": 0.65},
            "warnings": warnings,
            "version": MODELS.fusion_version,
        },
        "explanations": {
            "tabular": tabular_summary,
            "nlp": nlp_summary,
            "tabular_top_features": [{"feature": feature, "value": value} for feature, value in shap_top],
            "nlp_topk": nlp_summary.get("topk", []),
            "fusion_rationale": fusion_rationale,
        },
        "final_risk_category": final.get("risk_level") or final.get("risk_category", "Unknown"),
        "final_risk_score": final.get(motive_key, {}).get("conf", 0.0)
        if isinstance(final.get(motive_key), dict)
        else 0.0,
        "combined_confidence": final.get(motive_key, {}).get("conf", 0.0)
        if isinstance(final.get(motive_key), dict)
        else 0.0,
        "model_version": MODELS.fusion_version,
    }


if __name__ == "__main__":
    tab = {"risk_level": "Medium", "motive": "Property-focused", "offender_type": "Organized"}
    shap = [("incident_severity", 0.3), ("prior_incidents", 0.2)]
    nlp = [
        {"label": "Property-focused", "prob": 0.5},
        {"label": "Opportunistic", "prob": 0.3},
        {"label": "Control", "prob": 0.2},
    ]
    out = late_fusion_predict(tab, shap, nlp)
    assert "final" in out
    assert "motive" in out["final"]
    assert out["final"]["motive"]["pred"] in ("Property-focused", "Opportunistic", "Control")
    assert "fusion_meta" in out
    assert "explanations" in out
    print("OK:", out["final"]["motive"])
