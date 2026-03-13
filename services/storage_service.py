from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config import PATHS, ensure_dirs


CaseRecord = Dict[str, Any]


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@st.cache_data(show_spinner=False)
def load_cases() -> List[CaseRecord]:
    """Return all stored cases from the JSONL file.

    The cache is automatically invalidated when the underlying file changes.
    """
    ensure_dirs()
    path: Path = PATHS.cases_file
    if not path.exists():
        return []
    records: List[CaseRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines but do not break the app.
                continue
    return records


def _next_case_id(existing: List[CaseRecord]) -> str:
    prefix = "CASE-"
    numbers = []
    for rec in existing:
        cid = str(rec.get("case_id") or rec.get("id") or "")
        if cid.startswith(prefix):
            suffix = cid[len(prefix) :]
            if suffix.isdigit():
                numbers.append(int(suffix))
    next_num = (max(numbers) + 1) if numbers else 1
    return f"{prefix}{next_num:06d}"


def save_case(
    *,
    structured_input: Optional[Dict[str, Any]],
    narrative_input: Optional[str],
    tabular_output: Optional[Dict[str, Any]],
    nlp_output: Optional[Dict[str, Any]],
    fusion_output: Optional[Dict[str, Any]],
    evidence: Optional[List[Dict[str, Any]]],
    audit_meta: Optional[Dict[str, Any]] = None,
) -> CaseRecord:
    """Append a new case record to local storage and return it.

    This function is deliberately simple JSONL-based storage for the prototype.
    """
    ensure_dirs()
    existing = load_cases()
    case_id = _next_case_id(existing)

    record: CaseRecord = {
        "case_id": case_id,
        "created_at": _now_utc_iso(),
        "structured_input": structured_input or {},
        "narrative_input": narrative_input or "",
        "tabular_output": tabular_output or {},
        "nlp_output": nlp_output or {},
        "fusion_output": fusion_output or {},
        "evidence": evidence or [],
        "audit_meta": audit_meta or {},
    }

    path: Path = PATHS.cases_file
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Bust cache for subsequent reads.
    load_cases.clear()
    return record


def get_case_by_id(case_id: str) -> Optional[CaseRecord]:
    for rec in load_cases():
        if str(rec.get("case_id")) == str(case_id):
            return rec
    return None


def update_case_notes(case_id: str, reviewer_notes: str) -> Optional[CaseRecord]:
    """Update reviewer notes for a case in-place.

    For JSONL, we rewrite the file; acceptable for prototype-scale data.
    """
    ensure_dirs()
    path: Path = PATHS.cases_file
    if not path.exists():
        return None

    cases = load_cases()
    updated: Optional[CaseRecord] = None
    for rec in cases:
        if str(rec.get("case_id")) == str(case_id):
            rec.setdefault("reviewer_notes", "")
            rec["reviewer_notes"] = reviewer_notes
            updated = rec
            break

    if updated is None:
        return None

    with path.open("w", encoding="utf-8") as f:
        for rec in cases:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    load_cases.clear()
    return updated


__all__ = ["CaseRecord", "load_cases", "save_case", "get_case_by_id", "update_case_notes"]

