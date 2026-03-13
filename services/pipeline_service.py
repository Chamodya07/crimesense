"""Pipeline orchestration: tabular -> NLP -> fusion in one call."""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from services.fusion_service import late_fusion_predict
from services.nlp_service import predict_nlp_topk
from services.tabular_service import predict_tabular


def run_profile(
    case_dict: Dict[str, Any],
    narrative_text: Optional[str] = None,
    motive_key: Optional[str] = None,
    nlp_top_k: int = 5,
) -> Dict[str, Any]:
    """Execute the profile prediction pipeline.

    This is a thin orchestrator that calls the tabular, NLP and fusion
    services and returns the combined result.  It mirrors the behaviour
    described in the project specification.

    ``motive_key`` may be ``None`` which tells the pipeline to inspect the
    tabular output and choose the first of ``("motive", "primary_motive",
    "crime_motive")`` that is present.
    """
    tab_out = predict_tabular(case_dict or {})
    shap_top: List[Tuple[str, float]] = tab_out.get("shap_top", [])

    # determine motive key if caller didn't supply one
    if motive_key is None:
        flat = tab_out.get("pred") if "pred" in tab_out else tab_out
        for candidate in ("motive", "primary_motive", "crime_motive"):
            if candidate in flat:
                motive_key = candidate
                break
        if motive_key is None:
            motive_key = "motive"

    nlp_out = predict_nlp_topk(narrative_text or "", top_k=nlp_top_k)
    # ``nlp_out`` is either None or a dict with topk/pred/confidence
    topk_list = nlp_out.get("topk") if isinstance(nlp_out, dict) else None

    fused = late_fusion_predict(tab_out, shap_top, topk_list, motive_key=motive_key)

    # Attach raw outputs for storage/audit
    fused["_tabular_output"] = tab_out
    fused["_nlp_output"] = nlp_out or {}

    return fused


# backwards compatibility - older code may import run_pipeline
run_pipeline = run_profile


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _location_keyword(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    keywords = [
        "street",
        "residence",
        "apartment",
        "house",
        "business",
        "store",
        "park",
        "school",
        "vehicle",
        "sidewalk",
        "highway",
        "alley",
    ]
    for keyword in keywords:
        if keyword in text:
            return keyword
    return text


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = _text(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _safe_scalar(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        np = None  # type: ignore

    try:
        import pandas as pd  # type: ignore
    except Exception:  # noqa: BLE001
        pd = None  # type: ignore

    if np is not None and isinstance(value, np.generic):
        return value.item()
    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _simple_saved_similarity(current_case_dict: Dict[str, Any], saved_record: Dict[str, Any]) -> float:
    inputs = saved_record.get("inputs") or {}
    if not isinstance(inputs, dict):
        return 0.0

    score = 0.0

    cur_type = _text(current_case_dict.get("primary_type"))
    rec_type = _text(inputs.get("primary_type") or inputs.get("type"))
    if cur_type and rec_type and cur_type == rec_type:
        score += 2.0

    cur_weapon = _text(current_case_dict.get("weapon_desc"))
    rec_weapon = _text(inputs.get("weapon_desc") or inputs.get("weapon"))
    if cur_weapon and rec_weapon and cur_weapon == rec_weapon:
        score += 1.0

    cur_place = _location_keyword(current_case_dict.get("location_desc"))
    rec_place = _location_keyword(inputs.get("location_desc") or inputs.get("place"))
    if cur_place and rec_place and cur_place == rec_place:
        score += 1.0

    cur_night = _boolish(current_case_dict.get("is_night"))
    rec_night = _boolish(inputs.get("is_night"))
    if cur_night is not None and rec_night is not None and cur_night == rec_night:
        score += 0.5

    return score


def get_evidence_bundle(
    case_dict: Dict[str, Any],
    narrative_text: Optional[str],
    fused_output: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Collect RAG-index and saved-history evidence for the current case."""
    _ = fused_output

    rag_results: List[Dict[str, Any]] = []
    saved_results: List[Dict[str, Any]] = []
    warnings: List[str] = []

    try:
        from services.rag_service import retrieve_similar_cases

        rag_results, rag_message = retrieve_similar_cases(case_dict or {}, narrative_text or "", top_k=top_k)
        if rag_message:
            warnings.append(str(rag_message))
    except Exception as exc:  # noqa: BLE001
        rag_results = []
        warnings.append(f"RAG retrieval unavailable: {exc}")

    try:
        from services.firebase_service import list_history_records

        records = list_history_records(limit=200)
        scored: List[Dict[str, Any]] = []
        for rec in records:
            score = _simple_saved_similarity(case_dict or {}, rec or {})
            if score <= 0:
                continue
            inputs = rec.get("inputs") or {}
            outputs = rec.get("outputs") or rec.get("fused_output") or rec.get("fusion_output") or {}
            final = outputs.get("final", {}) if isinstance(outputs, dict) else {}
            motive_obj = final.get("motive", {})
            motive = motive_obj.get("pred") if isinstance(motive_obj, dict) else motive_obj
            motive_band = motive_obj.get("band") if isinstance(motive_obj, dict) else ""
            feedback = rec.get("feedback") or {}

            scored.append(
                {
                    "score": round(float(score), 3),
                    "saved_time": str(rec.get("created_at_local") or rec.get("created_at") or ""),
                    "record_id": rec.get("id", ""),
                    "type": inputs.get("primary_type") or inputs.get("type") or "",
                    "weapon": inputs.get("weapon_desc") or inputs.get("weapon") or "",
                    "place": inputs.get("location_desc") or inputs.get("place") or "",
                    "area": inputs.get("area") or inputs.get("city") or "",
                    "hour": _safe_scalar(inputs.get("hour")),
                    "is_night": _safe_scalar(inputs.get("is_night")),
                    "risk_level": _safe_scalar(
                        final.get("risk_level")
                        or final.get("risk_category")
                        or outputs.get("final_risk_category")
                        or ""
                    ),
                    "motive": _safe_scalar(motive or ""),
                    "motive_band": _safe_scalar(motive_band or ""),
                    "helpfulness": _safe_scalar(feedback.get("helpful") or feedback.get("helpfulness") or ""),
                    "feedback_text": _safe_scalar(feedback.get("text") or ""),
                    "_raw": rec,
                }
            )

        scored.sort(key=lambda item: (item.get("score", 0), item.get("saved_time", "")), reverse=True)
        saved_results = scored[: max(int(top_k), 0)]
        if not saved_results:
            warnings.append("No similar saved cases found.")
    except Exception as exc:  # noqa: BLE001
        saved_results = []
        warnings.append(f"Saved-history retrieval unavailable: {exc}")

    rag_message = "; ".join([w for w in warnings if "RAG" in w or "rag" in w]) or None
    saved_message = "; ".join([w for w in warnings if "saved" in w.lower() or "firebase" in w.lower()]) or None

    return {
        "rag_results": rag_results,
        "saved_similar": saved_results,
        "warnings": warnings,
        # Backward-compatible keys used by current UI/service call-sites.
        "rag": rag_results,
        "rag_message": rag_message,
        "saved": saved_results,
        "saved_message": saved_message,
    }


__all__ = ["run_profile", "run_pipeline", "get_evidence_bundle"]
