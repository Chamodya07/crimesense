from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None  # type: ignore

from services.rag_index import EMBED_MODEL_NAME
from services.utils_paths import get_rag_artifacts_dir


RAG_DIR = get_rag_artifacts_dir()
INDEX_PATH = RAG_DIR / "faiss.index"
CASES_PATH = RAG_DIR / "cases.csv"
DISPLAY_PRIORITY_COLUMNS = [
    "case_id",
    "type",
    "weapon",
    "place",
    "city",
    "area",
    "hour",
    "is_night",
    "victim_age",
    "victim_sex",
    "victim_race",
    "suspect_age",
    "suspect_sex",
    "suspect_race",
    "group_indicator",
    "prior_history",
    "law_category",
    "pd_description",
    "attempt_status",
    "status_desc",
    "arrest",
    "domestic",
]


def _get_cache_decorator():
    if st:
        return st.cache_resource(show_spinner=False)
    return lru_cache()


@_get_cache_decorator()
def load_index() -> Tuple[faiss.Index, pd.DataFrame, SentenceTransformer]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing RAG artifact: {INDEX_PATH}")
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Missing RAG artifact: {CASES_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    cases_df = pd.read_csv(CASES_PATH)
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load embedding model '{EMBED_MODEL_NAME}'. "
            "Ensure internet access on first run so the model can be downloaded."
        ) from exc
    return index, cases_df, model


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _jsonish(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _query_primary_type(case_dict: Dict[str, Any]) -> str:
    return _field_text(case_dict.get("primary_type") or case_dict.get("type"))


def _query_text_from_profile(case_dict: Dict[str, Any]) -> str:
    primary_type = _query_primary_type(case_dict)
    weapon_desc = _field_text(case_dict.get("weapon_desc"))
    location_desc = _field_text(case_dict.get("location_desc"))
    parts: List[str] = []
    if primary_type:
        parts.append(f"TYPE: {primary_type}")
    if weapon_desc:
        parts.append(f"WEAPON: {weapon_desc}")
    if location_desc:
        parts.append(f"PLACE: {location_desc}")

    # Optional temporal hints: contribute when available, never required.
    hour = case_dict.get("hour")
    if hour not in (None, ""):
        parts.append(f"HOUR: {hour}")
    is_night = case_dict.get("is_night")
    if isinstance(is_night, bool):
        parts.append("TIME: NIGHT" if is_night else "TIME: DAY")

    if not parts:
        return "TYPE: UNKNOWN"
    return "\n".join(parts)


def build_query_text(case_dict: Dict[str, Any]) -> str:
    """Public helper used by UI debug output to mirror retrieval query text."""
    return _query_text_from_profile(case_dict or {})


def retrieve_similar_cases(
    case_dict: Dict[str, Any],
    narrative_text: str,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], str | None]:
    # `narrative_text` is intentionally unused for this retrieval mode;
    # similarity is driven by structured profile fields only.
    _ = narrative_text
    if top_k <= 0:
        return [], None

    try:
        index, cases_df, model = load_index()
    except FileNotFoundError:
        return [], "RAG index artifacts are missing. Build the local index to enable similar case retrieval."
    except Exception as exc:  # noqa: BLE001
        return [], f"RAG retrieval unavailable: {exc}"
    query_text = _query_text_from_profile(case_dict or {})

    query_emb = model.encode([query_text], convert_to_numpy=True)
    query_emb = np.asarray(query_emb, dtype="float32")
    faiss.normalize_L2(query_emb)

    q_type = (case_dict.get("primary_type") or case_dict.get("type") or "").strip().upper()
    q_place = (case_dict.get("location_desc") or case_dict.get("place") or "").strip().upper()
    top_n = min(max(int(top_k) * 50, 200), len(cases_df))
    scores, idxs = index.search(query_emb, top_n)
    candidates: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(cases_df):
            continue
        row = cases_df.iloc[int(idx)]
        item: Dict[str, Any] = {"score": float(score)}
        for col in DISPLAY_PRIORITY_COLUMNS:
            if col not in cases_df.columns:
                continue
            value = _jsonish(row.get(col, ""))
            if value in ("", None):
                continue
            item[col] = value

        # Include additional non-empty contextual columns that may exist in
        # custom local datasets without requiring code changes.
        for col in cases_df.columns:
            if col in {"case_text"} or col in item:
                continue
            value = _jsonish(row.get(col, ""))
            if value in ("", None):
                continue
            item[col] = value

        if "case_id" not in item:
            item["case_id"] = _jsonish(row.get("case_id", "")) or str(int(idx))
        candidates.append(item)

    if q_type:
        type_matches = [r for r in candidates if str(r.get("type", "")).strip().upper() == q_type]
        type_place_matches: List[Dict[str, Any]] = []
        if q_place:
            type_place_matches = [
                r for r in type_matches if q_place in str(r.get("place", "")).strip().upper()
            ]

        print(
            f"RAG q_type={q_type} q_place={q_place} "
            f"candidates={len(candidates)} type_matches={len(type_matches)} place_matches={len(type_place_matches)}"
        )

        if q_place and type_place_matches:
            return type_place_matches[: int(top_k)], None

        if type_matches:
            return type_matches[: int(top_k)], None

        return [], None

    return candidates[: int(top_k)], None


def clear_index_cache() -> None:
    if hasattr(load_index, "cache_clear"):
        load_index.cache_clear()  # type: ignore[attr-defined]
    elif hasattr(load_index, "clear"):
        load_index.clear()  # type: ignore[attr-defined]


__all__ = [
    "load_index",
    "retrieve_similar_cases",
    "build_query_text",
    "clear_index_cache",
    "RAG_DIR",
    "INDEX_PATH",
    "CASES_PATH",
]
