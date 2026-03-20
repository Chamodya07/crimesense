from __future__ import annotations

import re
from datetime import datetime, timezone

from services.firebase_service import init_firebase


def normalize_city_code(city: str) -> str:
    normalized = str(city or "").strip()
    lowered = normalized.lower()

    if "new york" in lowered:
        return "NYC"
    if "los angeles" in lowered or lowered == "la":
        return "LAC"

    letters = re.findall(r"[A-Za-z]", normalized)
    if len(letters) >= 3:
        return "".join(letters[:3]).upper()
    return "UNK"


def generate_case_id(city: str, dt_utc: datetime) -> str:
    try:
        from firebase_admin import firestore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("firebase-admin is required to generate case IDs.") from exc

    if not isinstance(dt_utc, datetime):
        raise TypeError("dt_utc must be a datetime")
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(timezone.utc)

    year = dt_utc.year
    city_code = normalize_city_code(city)
    db = init_firebase()
    counter_ref = db.collection("counters").document(f"case_{city_code}_{year}")
    transaction = db.transaction()
    fallback_id = f"{city_code}-{year}-XX"

    @firestore.transactional
    def _increment_case_counter(transaction, doc_ref):
        snapshot = doc_ref.get(transaction=transaction)
        current_seq = 0
        if snapshot.exists:
            payload = snapshot.to_dict() or {}
            current_seq = int(payload.get("seq") or 0)
        next_seq = current_seq + 1
        transaction.set(doc_ref, {"seq": next_seq}, merge=True)
        return next_seq

    try:
        seq = _increment_case_counter(transaction, counter_ref)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to generate case ID via Firestore transaction. Fallback {fallback_id} was not used."
        ) from exc
    return f"{city_code}-{year}-{seq:02d}"


__all__ = ["normalize_city_code", "generate_case_id"]
