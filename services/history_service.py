from __future__ import annotations

from pathlib import Path
from typing import Any


def _to_time_text(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
        try:
            return str(value.isoformat())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def make_json_safe(obj: object):
    try:
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        np = None  # type: ignore

    try:
        import pandas as pd  # type: ignore
    except Exception:  # noqa: BLE001
        pd = None  # type: ignore

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return make_json_safe(obj.to_dict())
        if isinstance(obj, pd.DataFrame):
            return make_json_safe(obj.to_dict(orient="records"))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            item_value = obj.item()
            if item_value is not obj:
                return make_json_safe(item_value)
        except Exception:  # noqa: BLE001
            pass

    return str(obj)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _base_ui_case(
    *,
    doc_id: str,
    case_id: str,
    created_raw: object,
    structured: dict[str, Any],
    outputs: dict[str, Any],
    similar_cases: list[Any],
    feedback: dict[str, Any],
    narrative_text: str,
    raw_record: dict[str, Any],
) -> dict[str, Any]:
    final = outputs.get("final", {}) if isinstance(outputs, dict) else {}
    motive_obj = final.get("motive", {})
    motive = motive_obj.get("pred") if isinstance(motive_obj, dict) else motive_obj
    created_text = _to_time_text(created_raw)
    created_display = created_text or "-"

    risk = (
        final.get("risk_level")
        or final.get("risk_category")
        or outputs.get("final_risk_category")
        or "-"
    )

    feedback_text = _clean_text(feedback.get("text"))
    helpfulness = feedback.get("helpful") or feedback.get("helpfulness")
    notes_parts: list[str] = []
    if feedback_text:
        notes_parts.append(f"Feedback: {feedback_text}")
    if helpfulness:
        notes_parts.append(f"Helpful: {helpfulness}")

    crime_title = structured.get("primary_type") or structured.get("crime_type") or "Unknown Crime Type"
    crime_title = _clean_text(crime_title)
    if crime_title.lower() in {"", "unknown", "none", "-", "nan"}:
        crime_title = "Unknown Crime Type"

    description_text = (
        _clean_text(narrative_text)
        or _clean_text(structured.get("narrative_text"))
        or _clean_text(structured.get("description"))
        or "No description provided."
    )
    description_preview = f"{description_text[:300]}..." if len(description_text) > 300 else description_text

    motive_band = motive_obj.get("band") if isinstance(motive_obj, dict) else ""
    motive_conf = motive_obj.get("conf") if isinstance(motive_obj, dict) else ""
    motive_line = ""
    if motive:
        motive_line = f"Motive: {motive}"
        extra = []
        if motive_band:
            extra.append(f"Band={motive_band}")
        if motive_conf not in ("", None):
            extra.append(f"Confidence={motive_conf}")
        if extra:
            motive_line = f"{motive_line} ({', '.join(extra)})"
        notes_parts.insert(0, motive_line)

    if similar_cases:
        notes_parts.append(f"Similar cases saved: {len(similar_cases)}")
    else:
        notes_parts.append("No similar cases saved.")

    return {
        "_selector_id": doc_id or case_id,
        "doc_id": doc_id,
        "id": str(case_id),
        "title": crime_title,
        "date": created_display,
        "victim": (
            structured.get("victim_name")
            or structured.get("victim_type")
            or structured.get("victim")
            or "-"
        ),
        "age": structured.get("victim_age") or structured.get("age") or structured.get("vict_age"),
        "gender": (
            structured.get("victim_gender")
            or structured.get("victim_sex")
            or structured.get("gender")
            or "-"
        ),
        "location": (
            structured.get("location_desc")
            or structured.get("location")
            or structured.get("place")
            or structured.get("area")
            or "-"
        ),
        "description": description_text,
        "description_preview": description_preview,
        "risk": str(risk),
        "notes": " | ".join(notes_parts) if notes_parts else "",
        "motive": motive_line or "-",
        "feedback_text": feedback_text or "-",
        "helpful": str(helpfulness) if helpfulness not in (None, "") else "-",
        "inputs_full": make_json_safe(structured),
        "outputs_full": make_json_safe(outputs),
        "similar_cases_full": make_json_safe(similar_cases),
        "feedback_full": make_json_safe(feedback),
        "created_at_local": created_display,
        "_raw": raw_record,
        "_raw_safe": make_json_safe(raw_record),
    }


def firebase_record_to_ui_case(rec: dict[str, Any]) -> dict[str, Any]:
    outputs = (
        rec.get("outputs")
        or rec.get("fused")
        or rec.get("fused_output")
        or rec.get("fusion_output")
        or {}
    )
    structured = rec.get("inputs") or rec.get("structured_input") or {}
    if not isinstance(structured, dict):
        structured = {}

    similar_cases = rec.get("rag_results") or rec.get("similar_cases") or []
    if not isinstance(similar_cases, list):
        similar_cases = []

    feedback = rec.get("feedback") or {}
    if not isinstance(feedback, dict):
        feedback = {}

    doc_id = _clean_text(rec.get("id") or rec.get("record_id"))
    case_id = _clean_text(rec.get("case_id") or rec.get("id") or rec.get("record_id"))

    return _base_ui_case(
        doc_id=doc_id,
        case_id=case_id,
        created_raw=rec.get("created_at_local") or rec.get("created_at") or rec.get("timestamp"),
        structured=structured,
        outputs=outputs if isinstance(outputs, dict) else {},
        similar_cases=similar_cases,
        feedback=feedback,
        narrative_text=_clean_text(rec.get("narrative_text")),
        raw_record=rec,
    )


def _storage_record_to_ui_case(rec: dict[str, Any]) -> dict[str, Any]:
    structured = rec.get("structured_input") or rec.get("inputs") or {}
    if not isinstance(structured, dict):
        structured = {}

    outputs = rec.get("fusion_output") or rec.get("outputs") or {}
    if not isinstance(outputs, dict):
        outputs = {}

    similar_cases = rec.get("evidence") or rec.get("rag_results") or []
    if not isinstance(similar_cases, list):
        similar_cases = []

    reviewer_notes = _clean_text(rec.get("reviewer_notes"))
    feedback = {"text": reviewer_notes} if reviewer_notes else {}

    case_id = _clean_text(rec.get("case_id") or rec.get("id"))
    return _base_ui_case(
        doc_id=case_id,
        case_id=case_id,
        created_raw=rec.get("created_at"),
        structured=structured,
        outputs=outputs,
        similar_cases=similar_cases,
        feedback=feedback,
        narrative_text=_clean_text(rec.get("narrative_input")),
        raw_record=rec,
    )


def load_history_cases(limit: int = 50) -> list[dict[str, Any]]:
    try:
        from services.firebase_service import list_history_records

        records = list_history_records(limit=limit)
        if records:
            return [firebase_record_to_ui_case(record) for record in records]
    except Exception:
        pass

    try:
        from services.storage_service import load_cases

        cases = load_cases()
    except Exception:
        return []

    sorted_cases = sorted(
        cases,
        key=lambda item: str(item.get("created_at") or item.get("created_at_local") or ""),
        reverse=True,
    )
    return [_storage_record_to_ui_case(case) for case in sorted_cases[:limit]]


__all__ = ["load_history_cases", "firebase_record_to_ui_case", "make_json_safe"]
