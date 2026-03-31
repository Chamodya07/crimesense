from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.auth_service import render_auth_status
from services.firebase_service import get_case_by_id, list_history_records
from services.export_service import build_profile_doc
from services.history_service import firebase_record_to_ui_case, load_history_cases


ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"


def inject_styles() -> None:
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def clear_search() -> None:
    st.session_state["history_search_input"] = ""
    st.session_state["selected_case_id"] = None


def _to_time_text(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
        try:
            return str(value.isoformat())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def seed_cases():
    """Backend-connected case loader."""
    raw_records = []
    try:
        raw_records = list_history_records(limit=50)
    except Exception:
        raw_records = []
    return raw_records, load_history_cases(limit=50)


def _format_case_label(case: dict) -> str:
    return f"Case ID: {case.get('id', '-')} | {case.get('title', 'Unknown')} | {case.get('date', '-')}"


def _normalize_dataset_group(city_value: object, dataset_value: object) -> str:
    city = str(city_value or "").strip().lower()
    dataset_id = str(dataset_value or "").strip().lower()

    if dataset_id == "nypd" or "nypd" in city or "new york" in city or "ny" in city:
        return "nypd"
    if dataset_id == "la" or "los angeles" in city or city == "la" or "la" in city:
        return "la"
    return "other"


def _case_dataset_group(case: dict) -> str:
    raw_record = case.get("_raw") if isinstance(case.get("_raw"), dict) else {}
    raw_inputs = raw_record.get("inputs") if isinstance(raw_record.get("inputs"), dict) else {}

    city_value = (
        raw_record.get("city")
        or raw_inputs.get("city")
        or (case.get("inputs_full") or {}).get("city")
        or ""
    )
    dataset_value = (
        raw_record.get("rag_dataset")
        or raw_record.get("dataset_id")
        or (raw_record.get("rag") or {}).get("dataset")
        or ""
    )
    return _normalize_dataset_group(city_value, dataset_value)


def _humanize_key(key: object) -> str:
    return str(key).replace("_", " ").strip().title()


def _format_display_value(value: object) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, dict):
        pred = value.get("pred") if isinstance(value, dict) else None
        band = value.get("band") if isinstance(value, dict) else None
        conf = value.get("conf") if isinstance(value, dict) else None
        if pred not in (None, ""):
            details = []
            if band not in (None, ""):
                details.append(f"Band: {band}")
            if conf not in (None, ""):
                details.append(f"Confidence: {_format_display_value(conf)}")
            if details:
                return f"{pred} ({', '.join(details)})"
            return str(pred)

        parts = []
        for sub_key, sub_value in value.items():
            formatted = _format_display_value(sub_value)
            if formatted == "-":
                continue
            parts.append(f"{_humanize_key(sub_key)}: {formatted}")
        return " | ".join(parts) if parts else "-"
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            formatted = _format_display_value(item)
            if formatted != "-":
                items.append(formatted)
        if not items:
            return "-"
        preview = items[:5]
        suffix = "..." if len(items) > 5 else ""
        return f"{', '.join(preview)}{suffix}"
    return str(value)


def _render_compact_section(title: str, data: object, *, exclude_keys: set[str] | None = None) -> None:
    st.markdown(f"### {title}")
    if not isinstance(data, dict) or not data:
        st.write("-")
        return

    exclude = exclude_keys or set()
    rows = []
    for key, value in data.items():
        if key in exclude:
            continue
        formatted = _format_display_value(value)
        if formatted == "-":
            continue
        rows.append({"Field": _humanize_key(key), "Value": formatted})

    if not rows:
        st.write("-")
        return

    st.table(
        {
            "Field": [row["Field"] for row in rows],
            "Value": [row["Value"] for row in rows],
        }
    )


def _render_rag_evidence(case: dict) -> None:
    st.markdown("### RAG evidence")
    rag_payload = case.get("rag_full") or {}
    if not isinstance(rag_payload, dict):
        rag_payload = {}

    rag_summary = rag_payload.get("summary") or {}
    rag_items = rag_payload.get("top_cases") or case.get("similar_cases_full") or []
    similar_saved_cases = rag_payload.get("similar_saved_cases") or []

    has_summary = isinstance(rag_summary, dict) and any(value not in (None, "", [], {}) for value in rag_summary.values())
    has_rag_items = isinstance(rag_items, list) and bool(rag_items)
    has_saved_items = isinstance(similar_saved_cases, list) and bool(similar_saved_cases)

    if not has_summary and not has_rag_items and not has_saved_items:
        st.write("No RAG evidence saved.")
        return

    if has_summary:
        summary_lines = []
        for key in (
            "evidence_strength",
            "most_common_type",
            "most_common_place",
            "most_common_area",
            "most_common_law_category",
            "most_common_attempt_status",
            "most_common_weapon",
            "most_common_victim_age",
            "most_common_victim_sex",
            "most_common_victim_race",
            "most_common_suspect_age",
            "most_common_suspect_sex",
            "most_common_suspect_race",
        ):
            value = rag_summary.get(key)
            formatted = _format_display_value(value)
            if formatted == "-":
                continue
            summary_lines.append(f"- {_humanize_key(key)}: {formatted}")
        if summary_lines:
            st.markdown("\n".join(summary_lines))

    try:
        import pandas as pd

        if has_rag_items:
            st.markdown("#### Top Similar Cases")
            df_rag = pd.DataFrame(rag_items)
            preferred = [
                "score",
                "case_id",
                "type",
                "place",
                "area",
                "law_category",
                "attempt_status",
                "weapon",
                "victim_age",
                "victim_sex",
                "victim_race",
                "suspect_age",
                "suspect_sex",
                "suspect_race",
            ]
            cols = [c for c in preferred if c in df_rag.columns]
            if not cols:
                cols = list(df_rag.columns)
            st.dataframe(df_rag.reindex(columns=cols), use_container_width=True, hide_index=True)

        if has_saved_items:
            st.markdown("#### Similar Saved Cases")
            df_saved = pd.DataFrame(similar_saved_cases)
            preferred_saved = [
                "score",
                "saved_time",
                "record_id",
                "type",
                "weapon",
                "place",
                "city",
                "area",
                "hour",
                "is_night",
                "suspect_age",
                "suspect_sex",
                "suspect_race",
                "group_indicator",
                "prior_history",
                "arrest",
                "domestic",
                "risk_level",
                "motive",
                "motive_confidence",
                "motive_band",
                "helpfulness",
                "feedback_text",
                "victim_age",
                "victim_gender",
                "victim_race",
            ]
            saved_cols = [c for c in preferred_saved if c in df_saved.columns]
            if not saved_cols:
                saved_cols = list(df_saved.columns)
            st.dataframe(df_saved.reindex(columns=saved_cols), use_container_width=True, hide_index=True)
    except Exception:
        st.write("Saved RAG evidence could not be rendered as a table.")


def _render_export_button(case: dict) -> None:
    safe_case_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(case.get("id") or "case")
    )
    button_col, _ = st.columns([0.22, 0.78])
    raw_record = case.get("_raw") if isinstance(case.get("_raw"), dict) else {}
    export_record = {
        "case_id": case.get("id") or raw_record.get("case_id") or safe_case_id,
        "timestamp": raw_record.get("created_at_local") or raw_record.get("created_at") or case.get("date"),
        "inputs": case.get("inputs_full") or raw_record.get("inputs") or {},
        "outputs": case.get("outputs_full") or raw_record.get("outputs") or {},
        "rag_summary": (case.get("rag_full") or {}).get("summary") or {},
        "rag_results": (case.get("rag_full") or {}).get("top_cases") or case.get("similar_cases_full") or [],
        "similar_saved_cases": (case.get("rag_full") or {}).get("similar_saved_cases") or [],
        "feedback": case.get("feedback_full") or raw_record.get("feedback") or {},
        "rag": case.get("rag_full") or raw_record.get("rag") or {},
        "narrative_text": raw_record.get("narrative_text") or case.get("description") or "",
    }
    export_error = None
    export_bytes = None
    try:
        export_bytes = build_profile_doc(export_record)
    except Exception as exc:  # noqa: BLE001
        export_error = str(exc)
    with button_col:
        if export_bytes is not None:
            st.download_button(
                "Export",
                data=export_bytes,
                file_name=f"{safe_case_id}_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        else:
            st.button(
                "Export",
                disabled=True,
                key=f"export_history_docx_disabled_{safe_case_id}",
                help=export_error or "Export unavailable",
            )


def render_case_details(case: dict, show_title: bool = True) -> None:
    def _display_value(value):
        if value is None:
            return "\u2014"
        text = str(value).strip()
        if text in {"", "-", "--", "None", "nan"}:
            return "\u2014"
        return text

    c = {k: v for k, v in case.items() if k != "_raw"}
    raw_record = case.get("_raw") if isinstance(case.get("_raw"), dict) else {}
    raw_inputs = raw_record.get("inputs") if isinstance(raw_record.get("inputs"), dict) else {}
    victim = raw_inputs.get("victim")
    if not isinstance(victim, dict):
        victim = raw_record.get("victim")
    if not isinstance(victim, dict):
        victim = {}

    victim_age = _display_value(victim.get("age") or raw_inputs.get("victim_age") or c.get("age"))
    victim_gender = _display_value(
        victim.get("gender") or raw_inputs.get("victim_gender") or raw_inputs.get("victim_sex") or c.get("gender")
    )
    victim_race = _display_value(victim.get("race") or raw_inputs.get("victim_race"))

    if show_title:
        st.subheader(c["title"])
        st.caption(f"Case ID: {c.get('id', '-')}")
    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Date:** {c['date']}")
    meta_cols[1].markdown(f"**Location:** {c['location']}")
    meta_cols[2].markdown(f"**Risk:** {c['risk']}")

    st.markdown("### Victim / profile info")
    info_cols = st.columns(3)
    info_cols[0].markdown(f"**Victim Age:** {victim_age}")
    info_cols[1].markdown(f"**Gender:** {victim_gender}")
    info_cols[2].markdown(f"**Race:** {victim_race}")

    _render_compact_section(
        "Inputs",
        c.get("inputs_full"),
        exclude_keys={
            "narrative_text",
            "description",
            "victim",
            "victim_name",
            "victim_age",
            "victim_gender",
            "victim_sex",
            "victim_race",
        },
    )

    st.markdown("### Description")
    st.write(c["description"])

    output_payload = c.get("outputs_full")
    if isinstance(output_payload, dict) and isinstance(output_payload.get("final"), dict):
        compact_outputs = dict(output_payload.get("final") or {})
        if "risk_level" not in compact_outputs and output_payload.get("final_risk_category") not in (None, ""):
            compact_outputs["risk_level"] = output_payload.get("final_risk_category")
        if output_payload.get("combined_confidence") not in (None, ""):
            compact_outputs["combined_confidence"] = output_payload.get("combined_confidence")
        if output_payload.get("model_version") not in (None, ""):
            compact_outputs["model_version"] = output_payload.get("model_version")
    else:
        compact_outputs = output_payload

    _render_compact_section(
        "Outputs",
        compact_outputs,
        exclude_keys={"explanations", "fusion_meta", "tabular", "nlp", "warnings"},
    )

    st.markdown("### Notes / recommendations")
    st.markdown(c.get("motive", "-"))
    st.markdown(f"**Feedback:** {c.get('feedback_text', '-')}")
    st.markdown(f"**Helpful:** {c.get('helpful', '-')}")

    _render_rag_evidence(c)


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Past Predictions",
        page_icon="P",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    render_auth_status("history")

    if "history_search_input" not in st.session_state:
        st.session_state["history_search_input"] = ""
    if "history_dataset_filter" not in st.session_state:
        st.session_state["history_dataset_filter"] = "All"
    if "selected_case_id" not in st.session_state:
        st.session_state["selected_case_id"] = None

    header_left, header_right = st.columns([3, 1.2])
    with header_left:
        st.title("Past Predicted Profiles")
    with header_right:
        s_col, f_col, x_col = st.columns([0.54, 0.30, 0.16], gap="small")
        with s_col:
            st.text_input(
                "Search cases",
                placeholder="Search by case ID or crime type",
                label_visibility="collapsed",
                key="history_search_input",
            )
        with f_col:
            st.selectbox(
                "Dataset filter",
                options=["All", "NYC (NYPD)", "LA"],
                label_visibility="collapsed",
                key="history_dataset_filter",
            )
        with x_col:
            st.button("X", help="Clear search and show all cases", on_click=clear_search, key="clear_history_search")

    raw_records, all_cases = seed_cases()
    cases = list(all_cases)
    query = st.session_state.get("history_search_input", "").strip().lower()
    if query:
        cases = [
            c
            for c in cases
            if query in c["id"].lower()
            or query in c["title"].lower()
        ]
    dataset_filter = st.session_state.get("history_dataset_filter", "All")
    if dataset_filter == "NYC (NYPD)":
        cases = [c for c in cases if _case_dataset_group(c) == "nypd"]
    elif dataset_filter == "LA":
        cases = [c for c in cases if _case_dataset_group(c) == "la"]
    cases_sorted = sorted(cases, key=lambda c: c["date"], reverse=True)

    selected_case = None
    if cases_sorted:
        selectable_ids = [c.get("_selector_id") for c in cases_sorted if c.get("_selector_id")]
        if st.session_state.get("selected_case_id") not in selectable_ids:
            st.session_state["selected_case_id"] = selectable_ids[0] if selectable_ids else None
        selected_selector_id = st.session_state.get("selected_case_id")
        selected_case = next(
            (c for c in cases_sorted if c.get("_selector_id") == selected_selector_id),
            None,
        )
    else:
        st.session_state["selected_case_id"] = None

    st.markdown("**Cases**")
    if not all_cases:
        st.info("No saved cases yet.")
    elif not cases_sorted:
        st.info("No cases match your search or filter.")
    else:
        selectable_ids = [case.get("_selector_id") for case in cases_sorted if case.get("_selector_id")]
        current_case_id = st.session_state.get("selected_case_id")
        current_index = selectable_ids.index(current_case_id) if current_case_id in selectable_ids else 0
        selected_selector_id = st.selectbox(
            "Saved cases",
            options=selectable_ids,
            index=current_index,
            format_func=lambda selector_id: _format_case_label(
                next(case for case in cases_sorted if case.get("_selector_id") == selector_id)
            ),
        )
        st.session_state["selected_case_id"] = selected_selector_id
        selected_case = next(
            (
                c
                for c in cases_sorted
                if c.get("_selector_id") == selected_selector_id
            ),
            selected_case,
        )

    if selected_case:
        try:
            latest_record = get_case_by_id(selected_case.get("id", ""))
            if isinstance(latest_record, dict):
                selected_case = firebase_record_to_ui_case(latest_record)
        except Exception:
            pass

    if selected_case:
        _render_export_button(selected_case)
        render_case_details(selected_case)
    elif cases:
        st.info("Select a case from the dropdown to view its details.")
    else:
        st.info("No saved cases yet.")


if __name__ == "__main__":
    main()
