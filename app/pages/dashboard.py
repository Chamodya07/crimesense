from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from components.page_cards import render_page_cards
from services.auth_service import render_auth_status
from services.history_service import load_history_cases

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"


def inject_styles() -> None:
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _clean_text(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    if text in {"-", "None", "nan"}:
        return default
    return text or default


def _case_dataframe(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": _clean_text(case.get("id"), "-"),
                "crime_type": _clean_text(case.get("title"), "Unknown"),
                "location": _clean_text(case.get("location"), "-"),
                "risk": _clean_text(case.get("risk"), "Unknown"),
                "helpful": _clean_text(case.get("helpful")),
                "feedback_text": _clean_text(case.get("feedback_text")),
                "created_at": case.get("created_at_local") or case.get("date"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["created_ts"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["day"] = df["created_ts"].dt.date
    return df


def summary_row(df: pd.DataFrame) -> None:
    now = pd.Timestamp.utcnow()
    recent_cutoff = now - pd.Timedelta(days=7)
    previous_cutoff = now - pd.Timedelta(days=14)

    recent_cases = int((df["created_ts"] >= recent_cutoff).sum()) if "created_ts" in df else 0
    previous_cases = int(
        ((df["created_ts"] >= previous_cutoff) & (df["created_ts"] < recent_cutoff)).sum()
    ) if "created_ts" in df else 0
    delta = recent_cases - previous_cases
    delta_text = f"{delta:+d} vs prev 7d"

    high_risk = int(df["risk"].str.contains("high|critical|severe", case=False, na=False).sum()) if not df.empty else 0
    distinct_types = int(df["crime_type"].replace({"": "Unknown"}).nunique()) if not df.empty else 0

    cols = st.columns(4)
    cols[0].metric("Saved profiles", len(df))
    cols[1].metric("Cases in last 7 days", recent_cases, delta_text)
    cols[2].metric("Distinct crime types", distinct_types)
    cols[3].metric("High-risk cases", high_risk)


def render_timeline(df: pd.DataFrame) -> None:
    st.markdown("**Cases saved over time**")
    if df.empty or df["created_ts"].isna().all():
        st.info("No timestamped history records available yet.")
        return

    trend = (
        df.dropna(subset=["created_ts"])
        .groupby("day", as_index=False)
        .size()
        .rename(columns={"size": "Cases"})
        .sort_values("day")
    )
    st.line_chart(trend.set_index("day"))


def render_crime_type_chart(df: pd.DataFrame) -> None:
    st.markdown("**Most common crime types**")
    if df.empty:
        st.info("No case history available.")
        return

    top_types = (
        df.groupby("crime_type", as_index=False)
        .size()
        .rename(columns={"size": "Cases"})
        .sort_values("Cases", ascending=False)
        .head(8)
    )
    st.bar_chart(top_types.set_index("crime_type"))


def render_risk_chart(df: pd.DataFrame) -> None:
    st.markdown("**Risk distribution**")
    if df.empty:
        st.info("No risk outputs available.")
        return

    risks = (
        df.groupby("risk", as_index=False)
        .size()
        .rename(columns={"size": "Cases"})
        .sort_values("Cases", ascending=False)
    )
    st.bar_chart(risks.set_index("risk"))


def render_location_table(df: pd.DataFrame) -> None:
    st.markdown("**Top locations**")
    if df.empty:
        st.info("No saved locations available.")
        return

    locations = (
        df.groupby("location", as_index=False)
        .size()
        .rename(columns={"size": "Cases"})
        .sort_values("Cases", ascending=False)
        .head(10)
    )
    st.dataframe(locations, hide_index=True, use_container_width=True)


def render_recent_cases(df: pd.DataFrame) -> None:
    st.markdown("**Recent saved cases**")
    if df.empty:
        st.info("No saved cases available.")
        return

    recent = (
        df.sort_values("created_ts", ascending=False, na_position="last")
        .head(10)
        .rename(
            columns={
                "case_id": "Case ID",
                "crime_type": "Crime Type",
                "location": "Location",
                "risk": "Risk",
                "created_at": "Saved At",
            }
        )[["Case ID", "Crime Type", "Location", "Risk", "Saved At"]]
    )
    st.dataframe(recent, hide_index=True, use_container_width=True)


def render_feedback(df: pd.DataFrame) -> None:
    st.markdown("**Feedback activity**")
    if df.empty:
        st.info("No case feedback captured yet.")
        return

    feedback_rows = df[(df["feedback_text"] != "") | (df["helpful"] != "")]
    if feedback_rows.empty:
        st.info("Saved cases do not include feedback yet.")
        return

    feedback_summary = (
        feedback_rows.assign(helpful_bucket=feedback_rows["helpful"].replace({"": "Not answered", "-": "Not answered"}))
        .groupby("helpful_bucket", as_index=False)
        .size()
        .rename(columns={"size": "Responses", "helpful_bucket": "Helpfulness"})
        .sort_values("Responses", ascending=False)
    )
    st.bar_chart(feedback_summary.set_index("Helpfulness"))

    sample_feedback = feedback_rows[feedback_rows["feedback_text"] != ""][["case_id", "feedback_text"]].head(5)
    if not sample_feedback.empty:
        st.caption("Latest feedback")
        st.dataframe(
            sample_feedback.rename(columns={"case_id": "Case ID", "feedback_text": "Feedback"}),
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Dashboard",
        page_icon="DB",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    render_auth_status("dashboard")

    st.title("Dashboard")
    st.caption("Live overview of saved prediction history and operational activity.")
    render_page_cards("dashboard", title="")

    cases = load_history_cases(limit=200)
    df = _case_dataframe(cases)

    summary_row(df)

    top = st.columns((1.4, 1, 1))
    with top[0]:
        with st.container(border=True):
            render_timeline(df)
    with top[1]:
        with st.container(border=True):
            render_crime_type_chart(df)
    with top[2]:
        with st.container(border=True):
            render_risk_chart(df)

    bottom = st.columns((1, 1.2))
    with bottom[0]:
        with st.container(border=True):
            render_location_table(df)
        with st.container(border=True):
            render_feedback(df)
    with bottom[1]:
        with st.container(border=True):
            render_recent_cases(df)


if __name__ == "__main__":
    main()
