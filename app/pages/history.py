from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import streamlit as st

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.auth_service import render_auth_status
from services.history_service import load_history_cases

try:
    from docx import Document

    DOCX_AVAILABLE = True
except ImportError:
    Document = None
    DOCX_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    PDF_AVAILABLE = True
except ImportError:
    letter = None
    getSampleStyleSheet = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    PDF_AVAILABLE = False


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
    return load_history_cases(limit=50)


def _format_case_label(case: dict) -> str:
    return f"Case ID: {case.get('id', '-')} | {case.get('title', 'Unknown')} | {case.get('date', '-')}"


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
    rag_items = case.get("similar_cases_full") or []
    if not isinstance(rag_items, list) or not rag_items:
        st.write("No RAG evidence saved.")
        return

    try:
        import pandas as pd

        df_rag = pd.DataFrame(rag_items)
        preferred = [
            "score",
            "case_id",
            "type",
            "place",
            "area",
            "risk_level",
            "motive",
            "victim_age",
            "victim_sex",
            "suspect_age",
            "suspect_sex",
        ]
        cols = [c for c in preferred if c in df_rag.columns]
        if not cols:
            cols = list(df_rag.columns)
        st.dataframe(df_rag.reindex(columns=cols), use_container_width=True, hide_index=True)
    except Exception:
        st.write("Saved RAG evidence could not be rendered as a table.")


def _render_export_button(case: dict) -> None:
    safe_case_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(case.get("id") or "case")
    )
    button_col, _ = st.columns([0.22, 0.78])
    if DOCX_AVAILABLE:
        with button_col:
            st.download_button(
                "Export DOCX",
                data=build_case_docx(case),
                file_name=f"{safe_case_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        return

    if PDF_AVAILABLE:
        with button_col:
            st.download_button(
                "Export PDF",
                data=build_case_pdf(case),
                file_name=f"{safe_case_id}.pdf",
                mime="application/pdf",
            )
        return

    with button_col:
        st.download_button(
            "Export RTF",
            data=build_case_rtf(case),
            file_name=f"{safe_case_id}.rtf",
            mime="application/rtf",
        )


def build_case_docx(case: dict) -> bytes:
    """Create a simple Word document for a case export."""
    if not DOCX_AVAILABLE or Document is None:
        raise RuntimeError("python-docx is not installed")
    doc = Document()
    doc.add_heading(case["title"], level=1)

    doc.add_paragraph(f"Case ID: {case['id']}")
    doc.add_paragraph(f"Date: {case['date']}")
    doc.add_paragraph(f"Location: {case['location']}")
    doc.add_paragraph(f"Risk: {case['risk']}")

    doc.add_heading("Victim / profile info", level=2)
    doc.add_paragraph(f"Victim: {case['victim']}")
    age_value = case.get("age", "--") if case.get("age") is not None else "--"
    doc.add_paragraph(f"Age: {age_value}")
    doc.add_paragraph(f"Gender: {case.get('gender', '--')}")

    doc.add_heading("Description", level=2)
    doc.add_paragraph(case["description"])

    doc.add_heading("Notes / recommendations", level=2)
    doc.add_paragraph(case["notes"])

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def build_case_rtf(case: dict) -> bytes:
    """Create a Word-compatible RTF export without external dependencies."""

    def safe(value: object) -> str:
        text = str(value) if value is not None else ""
        text = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", r"\line ")
        return text

    age_value = case.get("age", "--") if case.get("age") is not None else "--"
    sections = [
        (r"\b " + safe(case.get("title", "Case Export")) + r"\b0", ""),
        ("Case ID:", safe(case.get("id", "--"))),
        ("Date:", safe(case.get("date", "--"))),
        ("Location:", safe(case.get("location", "--"))),
        ("Risk:", safe(case.get("risk", "--"))),
        (r"\b Victim / profile info\b0", ""),
        ("Victim:", safe(case.get("victim", "--"))),
        ("Age:", safe(age_value)),
        ("Gender:", safe(case.get("gender", "--"))),
        (r"\b Description\b0", ""),
        ("", safe(case.get("description", ""))),
        (r"\b Notes / recommendations\b0", ""),
        ("", safe(case.get("notes", ""))),
    ]

    lines = [r"{\rtf1\ansi\deff0"]
    for label, value in sections:
        if label and value:
            lines.append(rf"{label} {value}\line ")
        elif label:
            lines.append(rf"{label}\line ")
        else:
            lines.append(rf"{value}\line ")
    lines.append("}")
    return "".join(lines).encode("utf-8")


def build_case_pdf(case: dict) -> bytes:
    """Create a simple PDF document for a case export."""
    if not PDF_AVAILABLE or SimpleDocTemplate is None:
        raise RuntimeError("reportlab is not installed")

    def safe(value: object) -> str:
        return escape(str(value)) if value is not None else ""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=str(case.get("title", "Case Export")))
    styles = getSampleStyleSheet()
    story = [
        Paragraph(safe(case.get("title", "Case Export")), styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"<b>Case ID:</b> {safe(case.get('id', '--'))}", styles["BodyText"]),
        Paragraph(f"<b>Date:</b> {safe(case.get('date', '--'))}", styles["BodyText"]),
        Paragraph(f"<b>Location:</b> {safe(case.get('location', '--'))}", styles["BodyText"]),
        Paragraph(f"<b>Risk:</b> {safe(case.get('risk', '--'))}", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Victim / profile info", styles["Heading2"]),
        Paragraph(f"<b>Victim:</b> {safe(case.get('victim', '--'))}", styles["BodyText"]),
        Paragraph(
            f"<b>Age:</b> {safe(case.get('age', '--') if case.get('age') is not None else '--')}",
            styles["BodyText"],
        ),
        Paragraph(f"<b>Gender:</b> {safe(case.get('gender', '--'))}", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Description", styles["Heading2"]),
        Paragraph(safe(case.get("description", "")), styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Notes / recommendations", styles["Heading2"]),
        Paragraph(safe(case.get("notes", "")), styles["BodyText"]),
    ]
    doc.build(story)
    return buffer.getvalue()


def render_case_details(case: dict, show_title: bool = True) -> None:
    c = {k: v for k, v in case.items() if k != "_raw"}
    if show_title:
        st.subheader(c["title"])
        st.caption(f"Case ID: {c.get('id', '-')}")
    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Date:** {c['date']}")
    meta_cols[1].markdown(f"**Location:** {c['location']}")
    meta_cols[2].markdown(f"**Risk:** {c['risk']}")

    st.markdown("### Victim / profile info")
    info_cols = st.columns(3)
    info_cols[0].markdown(f"**Victim:** {c['victim']}")
    info_cols[1].markdown(f"**Age:** {c.get('age','--') if c.get('age') is not None else '--'}")
    info_cols[2].markdown(f"**Gender:** {c.get('gender','--')}")

    _render_compact_section(
        "Inputs",
        c.get("inputs_full"),
        exclude_keys={"narrative_text", "description", "victim_name", "victim_age", "victim_gender", "victim_sex"},
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
    if "selected_case_id" not in st.session_state:
        st.session_state["selected_case_id"] = None

    header_left, header_right = st.columns([3, 1.2])
    with header_left:
        st.title("Past Predicted Profiles")
        st.caption("Review prior predictions; select a case to view its details.")
    with header_right:
        s_col, x_col = st.columns([0.82, 0.18], gap="small")
        with s_col:
            st.text_input(
                "Search cases",
                placeholder="Search by case ID or crime type",
                label_visibility="collapsed",
                key="history_search_input",
            )
        with x_col:
            st.button("X", help="Clear search and show all cases", on_click=clear_search, key="clear_history_search")

    cases = seed_cases()
    query = st.session_state.get("history_search_input", "").strip().lower()
    if query:
        cases = [
            c
            for c in cases
            if query in c["id"].lower()
            or query in c["title"].lower()
        ]
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
    if not cases_sorted:
        st.info("No cases match your search.")
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
        _render_export_button(selected_case)
        render_case_details(selected_case)
    elif cases:
        st.info("Select a case from the dropdown to view its details.")
    else:
        st.info("No saved cases are available.")


if __name__ == "__main__":
    main()
