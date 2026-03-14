from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import streamlit as st

from services.firebase_service import FirebaseConfigError, init_firebase, make_json_safe


_AUDIT_WARNING_KEY = "_audit_service_warning"


def get_current_user_label(default: str = "unknown") -> str:
    payload = st.session_state.get("auth_user")
    if isinstance(payload, dict):
        username = str(payload.get("username", "")).strip()
        if username:
            return username
    return default


def _warn_once(message: str) -> None:
    if st.session_state.get(_AUDIT_WARNING_KEY) == message:
        return
    st.session_state[_AUDIT_WARNING_KEY] = message
    st.warning(message)


def _read_timestamp(record: dict[str, Any]) -> datetime | None:
    value = record.get("ts_utc") or record.get("ts_local")
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt_value = value
    else:
        try:
            dt_value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


def _record_in_range(record: dict[str, Any], start_date: date | None, end_date: date | None) -> bool:
    if start_date is None and end_date is None:
        return True
    ts_value = _read_timestamp(record)
    if ts_value is None:
        return False
    ts_date = ts_value.date()
    if start_date is not None and ts_date < start_date:
        return False
    if end_date is not None and ts_date > end_date:
        return False
    return True


def log_event(
    action: str,
    user: str,
    case_id: str | None = None,
    model_ver: str | None = None,
    data_ver: str | None = None,
    page: str | None = None,
    meta: dict[str, Any] | None = None,
) -> bool:
    try:
        from firebase_admin import firestore
    except Exception:
        return False

    try:
        db = init_firebase()
    except FirebaseConfigError:
        return False
    except Exception:
        return False

    event = make_json_safe(
        {
            "ts_utc": firestore.SERVER_TIMESTAMP,
            "ts_local": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "user": str(user or "unknown"),
            "action": str(action or "").strip().lower(),
            "case_id": str(case_id).strip() if case_id not in (None, "") else None,
            "model_ver": str(model_ver).strip() if model_ver not in (None, "") else None,
            "data_ver": str(data_ver).strip() if data_ver not in (None, "") else None,
            "page": str(page or "").strip() or None,
            "meta": meta or {},
        }
    )
    try:
        db.collection("audit_events").add(event)
        return True
    except Exception:
        return False


def query_events(
    user: str | None = None,
    actions: list[str] | None = None,
    case_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    try:
        from firebase_admin import firestore

        db = init_firebase()
        docs = (
            db.collection("audit_events")
            .order_by("ts_utc", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
    except FirebaseConfigError as exc:
        _warn_once(str(exc))
        return []
    except Exception as exc:
        _warn_once(f"Audit log unavailable: {exc}")
        return []

    items = [{"id": doc.id, **make_json_safe(doc.to_dict() or {})} for doc in docs]

    normalized_user = str(user or "").strip()
    normalized_actions = {str(action).strip().lower() for action in (actions or []) if str(action).strip()}
    normalized_case = str(case_id or "").strip().lower()

    filtered: list[dict[str, Any]] = []
    for item in items:
        item_user = str(item.get("user") or "").strip()
        item_action = str(item.get("action") or "").strip().lower()
        item_case = str(item.get("case_id") or "").strip()

        if normalized_user and normalized_user != "All users" and item_user != normalized_user:
            continue
        if normalized_actions and item_action not in normalized_actions:
            continue
        if normalized_case and normalized_case not in item_case.lower():
            continue
        if not _record_in_range(item, start_date, end_date):
            continue

        filtered.append(item)

    filtered.sort(key=lambda record: _read_timestamp(record) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return filtered[:limit]


def list_distinct_users(limit: int = 200) -> list[str]:
    items = query_events(limit=limit)
    users = sorted({str(item.get("user") or "").strip() for item in items if str(item.get("user") or "").strip()})
    return users


__all__ = [
    "get_current_user_label",
    "log_event",
    "query_events",
    "list_distinct_users",
]
