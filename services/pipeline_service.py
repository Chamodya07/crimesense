"""Pipeline orchestration: tabular -> NLP -> fusion in one call."""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from services.fusion_service import late_fusion_predict
from services.nlp_service import predict_nlp_topk
from services.tabular_service import predict_tabular


def _feature_to_phrase(feat: str, value: Any) -> str:
    feature = str(feat or "").strip()
    if feature.startswith("primary_type_"):
        return f"Crime type: {feature.removeprefix('primary_type_')}"
    if feature.startswith("location_desc_"):
        return f"Location: {feature.removeprefix('location_desc_')}"
    if feature.startswith("weapon_desc_"):
        return f"Weapon: {feature.removeprefix('weapon_desc_')}"
    if feature == "is_night":
        return "Night-time incident"
    if feature == "hour":
        return f"Hour: {_safe_scalar(value)}"
    if feature == "domestic":
        return "Domestic incident flag"
    if feature in {"latitude", "longitude"}:
        return "Geographic location"
    return feature.replace("_", " ")


def _build_tabular_text_expl(explanations: Dict[str, Any]) -> Dict[str, Any]:
    rows = explanations.get("tabular_shap_top", []) or []
    if not isinstance(rows, list):
        return {}

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature", "") or "").strip()
        if not feature:
            continue
        value = _safe_scalar(row.get("value"))
        try:
            shap = float(_safe_scalar(row.get("shap", 0.0)) or 0.0)
        except Exception:  # noqa: BLE001
            shap = 0.0
        normalized_rows.append(
            {
                "feature": feature,
                "value": value,
                "shap": shap,
                "phrase": _feature_to_phrase(feature, value),
            }
        )

    if not normalized_rows:
        return {}

    sorted_rows = sorted(normalized_rows, key=lambda row: abs(row["shap"]), reverse=True)
    up_rows = [row for row in sorted_rows if row["shap"] > 0][:3]
    down_rows = [row for row in sorted_rows if row["shap"] < 0][:2]

    drivers_up = [f"{row['phrase']} (+{row['shap']:.2f} impact)" for row in up_rows]
    drivers_down = [f"{row['phrase']} ({row['shap']:.2f} impact)" for row in down_rows]

    up_phrases = [row["phrase"] for row in up_rows[:2]]
    down_phrase = down_rows[0]["phrase"] if down_rows else "no strong opposing factors"

    summary = (
        f"Main factors increasing the prediction: {', '.join(up_phrases) if up_phrases else 'none identified'}. "
        f"Factors decreasing it: {down_phrase}."
    )

    return {
        "summary": str(summary),
        "drivers_up": [str(item) for item in drivers_up],
        "drivers_down": [str(item) for item in drivers_down],
    }


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

    fused = late_fusion_predict(tab_out, shap_top, nlp_out, motive_key=motive_key)

    explanations = fused.get("explanations")
    if not isinstance(explanations, dict):
        explanations = {}
        fused["explanations"] = explanations

    tabular_explanations = tab_out.get("explanations") if isinstance(tab_out, dict) else {}
    if isinstance(tabular_explanations, dict):
        existing_rows = explanations.get("tabular_shap_top", []) or []
        tabular_rows = tabular_explanations.get("tabular_shap_top", []) or []
        if not existing_rows and tabular_rows:
            explanations["tabular_shap_top"] = tabular_rows

    if isinstance(nlp_out, dict):
        explanations["nlp_topk"] = nlp_out.get("topk", []) or []
        explanations["nlp_pred"] = nlp_out.get("pred", "") or ""
        explanations["nlp_confidence"] = float(nlp_out.get("confidence", 0.0) or 0.0)
    else:
        explanations["nlp_topk"] = []
        explanations["nlp_pred"] = ""
        explanations["nlp_confidence"] = 0.0

    # Attach raw outputs for storage/audit
    fused["_tabular_output"] = tab_out
    fused["_nlp_output"] = nlp_out or {}

    if explanations.get("tabular_shap_top") and not explanations.get("tabular_text_expl"):
        explanations["tabular_text_expl"] = _build_tabular_text_expl(explanations)

    return fused


# backwards compatibility - older code may import run_pipeline
run_pipeline = run_profile


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize(value: Any) -> str:
    return _text(value)


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


def _hour_value(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except Exception:  # noqa: BLE001
        return None


def _saved_primary_type(saved_record: Dict[str, Any]) -> str:
    inputs = saved_record.get("inputs") or {}
    if isinstance(inputs, dict):
        return _normalize(inputs.get("primary_type"))
    return ""


def _current_primary_type(current_case_dict: Dict[str, Any]) -> str:
    return _normalize((current_case_dict or {}).get("primary_type"))


def _simple_saved_similarity(current_case_dict: Dict[str, Any], saved_record: Dict[str, Any]) -> float:
    inputs = saved_record.get("inputs") or {}
    if not isinstance(inputs, dict):
        return 0.0

    score = 0.0

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
    saved_debug: Dict[str, Any] = {
        "cur_type_raw": "",
        "cur_type": "",
        "saved_types_sample": [],
        "same_type_records": 0,
        "total_records": 0,
    }

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
        cur_type_raw = (case_dict or {}).get("primary_type")
        cur_type = _current_primary_type(case_dict or {})
        saved_types_sample = [_saved_primary_type(rec or {}) for rec in records[:10]]
        saved_debug = {
            "cur_type_raw": cur_type_raw,
            "cur_type": cur_type,
            "saved_types_sample": saved_types_sample,
            "same_type_records": 0,
            "total_records": len(records),
        }

        if not cur_type or cur_type in {"unknown", ""}:
            saved_results = []
            warnings.append("No similar saved cases found.")
            return {
                "rag_results": rag_results,
                "saved_similar": saved_results,
                "warnings": warnings,
                "rag": rag_results,
                "rag_message": "; ".join([w for w in warnings if "RAG" in w or "rag" in w]) or None,
                "saved": saved_results,
                "saved_message": "; ".join([w for w in warnings if "saved" in w.lower() or "firebase" in w.lower()]) or None,
                "saved_debug": saved_debug,
            }

        same_type_records = [
            rec
            for rec in records
            if _saved_primary_type(rec or {}) and _saved_primary_type(rec or {}) == cur_type
        ]
        saved_debug["same_type_records"] = len(same_type_records)
        scored: List[Dict[str, Any]] = []
        for rec in same_type_records:
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
                    "type": inputs.get("primary_type") or "",
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
        "saved_debug": saved_debug,
    }


__all__ = ["run_profile", "run_pipeline", "get_evidence_bundle"]
