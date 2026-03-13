from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from services.utils_paths import get_project_root


class FirebaseConfigError(RuntimeError):
    pass


def _load_service_account_dict() -> Dict[str, Any]:
    if "FIREBASE_SERVICE_ACCOUNT_JSON" in st.secrets:
        raw = st.secrets["FIREBASE_SERVICE_ACCOUNT_JSON"]
        if isinstance(raw, dict):
            return dict(raw)
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise FirebaseConfigError("Invalid FIREBASE_SERVICE_ACCOUNT_JSON in Streamlit secrets.") from exc

    if "FIREBASE_SERVICE_ACCOUNT_PATH" in st.secrets:
        raw_path = Path(str(st.secrets["FIREBASE_SERVICE_ACCOUNT_PATH"]))
        if not raw_path.is_absolute():
            raw_path = get_project_root() / raw_path
        if not raw_path.exists():
            raise FirebaseConfigError(f"FIREBASE_SERVICE_ACCOUNT_PATH not found: {raw_path}")
        try:
            return json.loads(raw_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FirebaseConfigError(f"Invalid JSON in service account file: {raw_path}") from exc

    raise FirebaseConfigError("Firebase not configured. Add service account JSON to Streamlit secrets.")


def make_json_safe(obj: Any) -> Any:
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


@st.cache_resource(show_spinner=False)
def init_firebase():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError as exc:
        raise FirebaseConfigError(
            "firebase-admin is not installed. Add it to requirements and install dependencies."
        ) from exc

    service_account = _load_service_account_dict()
    if firebase_admin._apps:
        app = firebase_admin.get_app()
    else:
        app = firebase_admin.initialize_app(credentials.Certificate(service_account))
    return firestore.client(app)


def save_history_record(record: Dict[str, Any]) -> str:
    try:
        from firebase_admin import firestore
    except Exception as exc:  # noqa: BLE001
        raise FirebaseConfigError(
            "firebase-admin is not installed. Add it to requirements and install dependencies."
        ) from exc

    db = init_firebase()
    payload = make_json_safe(record or {})
    payload.setdefault("created_at_local", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    payload["created_at"] = firestore.SERVER_TIMESTAMP

    _, doc_ref = db.collection("history").add(payload)
    return doc_ref.id


def save_case_by_id(case_id: str, record: Dict[str, Any]) -> None:
    try:
        from firebase_admin import firestore
    except Exception as exc:  # noqa: BLE001
        raise FirebaseConfigError(
            "firebase-admin is not installed. Add it to requirements and install dependencies."
        ) from exc

    normalized_case_id = str(case_id or "").strip()
    if not normalized_case_id:
        raise ValueError("case_id is required for save_case_by_id")

    db = init_firebase()
    payload = make_json_safe(record or {})
    payload["case_id"] = normalized_case_id
    payload.setdefault("created_at_local", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    payload["created_at"] = firestore.SERVER_TIMESTAMP
    db.collection("history").document(normalized_case_id).set(payload, merge=True)


def get_case_by_id(case_id: str) -> Dict[str, Any] | None:
    normalized_case_id = str(case_id or "").strip()
    if not normalized_case_id:
        return None

    db = init_firebase()
    doc = db.collection("history").document(normalized_case_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    return {"id": doc.id, **data}


def list_case_ids(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        from firebase_admin import firestore
    except Exception as exc:  # noqa: BLE001
        raise FirebaseConfigError(
            "firebase-admin is not installed. Add it to requirements and install dependencies."
        ) from exc

    db = init_firebase()
    try:
        docs = (
            db.collection("history")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
    except Exception:
        docs = db.collection("history").stream()

    items: List[Dict[str, Any]] = []
    for doc in docs:
        payload = doc.to_dict() or {}
        case_id = payload.get("case_id") or doc.id
        created = payload.get("created_at_local") or payload.get("created_at")
        inputs = payload.get("inputs") or {}
        items.append(
            {
                "case_id": str(case_id),
                "created_at_local": _to_time_text(created),
                "primary_type": inputs.get("primary_type") if isinstance(inputs, dict) else "",
            }
        )
    items.sort(key=lambda x: str(x.get("created_at_local") or ""), reverse=True)
    return items[:limit]


def list_cases(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        from firebase_admin import firestore
    except Exception as exc:  # noqa: BLE001
        raise FirebaseConfigError(
            "firebase-admin is not installed. Add it to requirements and install dependencies."
        ) from exc

    db = init_firebase()
    try:
        docs = (
            db.collection("history")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
    except Exception:
        docs = db.collection("history").stream()

    records: List[Dict[str, Any]] = []
    for doc in docs:
        payload = doc.to_dict() or {}
        case_id = payload.get("case_id") or doc.id
        records.append({"id": doc.id, "case_id": case_id, **payload})

    records.sort(
        key=lambda r: str(r.get("created_at_local") or r.get("created_at") or ""),
        reverse=True,
    )
    return records[:limit]


def list_history_records(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        from firebase_admin import firestore
    except Exception as exc:  # noqa: BLE001
        raise FirebaseConfigError(
            "firebase-admin is not installed. Add it to requirements and install dependencies."
        ) from exc

    db = init_firebase()
    try:
        docs = (
            db.collection("history")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
    except Exception:
        docs = db.collection("history").stream()

    records: List[Dict[str, Any]] = []
    for doc in docs:
        item = doc.to_dict() or {}
        records.append({"id": doc.id, **item})
    # Fallback ordering if query ordering is unavailable.
    records.sort(
        key=lambda r: str(r.get("created_at_local") or r.get("created_at") or ""),
        reverse=True,
    )
    return records[:limit]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _location_keyword(value: Any) -> str:
    text = _normalize_text(value)
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


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = _normalize_text(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def compute_simple_similarity(current_case_dict: Dict[str, Any], saved_record: Dict[str, Any]) -> float:
    inputs = saved_record.get("inputs") or {}
    if not isinstance(inputs, dict):
        return 0.0

    score = 0.0
    current_type = _normalize_text(current_case_dict.get("primary_type"))
    saved_type = _normalize_text(inputs.get("primary_type") or inputs.get("type"))
    if current_type and saved_type and current_type == saved_type:
        score += 2.0

    current_weapon = _normalize_text(current_case_dict.get("weapon_desc"))
    saved_weapon = _normalize_text(inputs.get("weapon_desc") or inputs.get("weapon"))
    if current_weapon and saved_weapon and current_weapon == saved_weapon:
        score += 1.0

    current_location = _location_keyword(current_case_dict.get("location_desc"))
    saved_location = _location_keyword(inputs.get("location_desc") or inputs.get("place"))
    if current_location and saved_location and current_location == saved_location:
        score += 1.0

    current_night = _as_bool(current_case_dict.get("is_night"))
    saved_night = _as_bool(inputs.get("is_night"))
    if current_night is not None and saved_night is not None and current_night == saved_night:
        score += 0.5

    return score


def _to_time_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
        try:
            return str(value.isoformat())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _extract_outputs(saved_record: Dict[str, Any]) -> Dict[str, Any]:
    outputs = (
        saved_record.get("outputs")
        or saved_record.get("fused")
        or saved_record.get("fused_output")
        or saved_record.get("fusion_output")
        or {}
    )
    return outputs if isinstance(outputs, dict) else {}


def find_similar_saved_cases(
    current_case_dict: Dict[str, Any],
    top_k: int = 5,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []

    try:
        records = list_history_records(limit=limit)
    except Exception:
        return []

    scored: List[Dict[str, Any]] = []
    for rec in records:
        sim_score = compute_simple_similarity(current_case_dict or {}, rec or {})
        if sim_score <= 0:
            continue

        inputs = rec.get("inputs") or {}
        outputs = _extract_outputs(rec)
        final = outputs.get("final", {}) if isinstance(outputs, dict) else {}
        motive_obj = final.get("motive", {})
        motive = motive_obj.get("pred") if isinstance(motive_obj, dict) else motive_obj
        motive_band = motive_obj.get("band") if isinstance(motive_obj, dict) else ""
        risk_level = final.get("risk_level") or final.get("risk_category") or outputs.get("final_risk_category")
        feedback = rec.get("feedback") or {}

        scored.append(
            {
                "score": round(float(sim_score), 3),
                "saved_time": _to_time_text(rec.get("created_at_local") or rec.get("created_at")),
                "record_id": rec.get("id", ""),
                "case_id": inputs.get("case_id") or rec.get("id", ""),
                "type": inputs.get("primary_type") or inputs.get("type") or "",
                "weapon": inputs.get("weapon_desc") or inputs.get("weapon") or "",
                "place": inputs.get("location_desc") or inputs.get("place") or "",
                "city": inputs.get("city") or "",
                "area": inputs.get("area") or inputs.get("city") or "",
                "hour": inputs.get("hour"),
                "is_night": inputs.get("is_night"),
                "victim_age": inputs.get("victim_age") or inputs.get("victim_age_group") or "",
                "victim_sex": inputs.get("victim_sex") or "",
                "victim_race": inputs.get("victim_race") or "",
                "suspect_age": inputs.get("suspect_age") or inputs.get("suspect_age_group") or "",
                "suspect_sex": inputs.get("suspect_sex") or "",
                "suspect_race": inputs.get("suspect_race") or "",
                "group_indicator": inputs.get("group_indicator") or inputs.get("group_involvement") or "",
                "prior_history": inputs.get("prior_history") or inputs.get("offender_experience") or "",
                "arrest": inputs.get("arrest"),
                "domestic": inputs.get("domestic"),
                "risk_level": risk_level or "",
                "motive": motive or "",
                "motive_confidence": motive_obj.get("conf") if isinstance(motive_obj, dict) else "",
                "motive_band": motive_band or "",
                "helpfulness": feedback.get("helpful") or feedback.get("helpfulness") or "",
                "feedback_text": feedback.get("text") or "",
                "_raw": rec,
            }
        )

    scored.sort(key=lambda item: (item.get("score", 0), item.get("saved_time", "")), reverse=True)
    return scored[:top_k]


__all__ = [
    "FirebaseConfigError",
    "init_firebase",
    "make_json_safe",
    "save_history_record",
    "save_case_by_id",
    "get_case_by_id",
    "list_case_ids",
    "list_cases",
    "list_history_records",
    "compute_simple_similarity",
    "find_similar_saved_cases",
]
