from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"


def inject_styles() -> None:
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _audit_from_storage() -> list[dict]:
    """Derive audit events from stored cases."""
    try:
        from services.storage_service import load_cases
        cases = load_cases()
        events = []
        for c in cases[:20]:
            meta = c.get("audit_meta") or {}
            events.append({
                "timestamp": c.get("created_at", ""),
                "user": "system",
                "action": "predict",
                "case_id": c.get("case_id", "—"),
                "model_version": meta.get("fusion_version", "—"),
                "data_version": meta.get("dataset_version", "—"),
                "status": "success",
                "channel": "UI",
                "ip": "—",
                "latency_ms": 0,
                "notes": f"Case {c.get('case_id')} saved.",
                "request_id": f"req_{c.get('case_id', '')}",
            })
        return events
    except Exception:
        return []


def seed_audit_log() -> list[dict]:
    """Backend-connected: demo events + events from stored cases."""
    from_storage = _audit_from_storage()
    demo = [
        {
            "timestamp": "2026-02-03T16:04:12Z",
            "user": "jlee",
            "action": "predict",
            "case_id": "CASE-2311",
            "model_version": "v1.8.2",
            "data_version": "2026.01.30",
            "status": "success",
            "channel": "UI",
            "ip": "10.4.22.19",
            "latency_ms": 842,
            "notes": "Ran offender profile inference for burglary pattern.",
            "request_id": "req_4f2a",
        },
        {
            "timestamp": "2026-02-03T15:12:41Z",
            "user": "asoto",
            "action": "save",
            "case_id": "CASE-2299",
            "model_version": "v1.8.2",
            "data_version": "2026.01.30",
            "status": "success",
            "channel": "UI",
            "ip": "10.4.18.6",
            "latency_ms": 214,
            "notes": "Saved investigator feedback with escalation flag.",
            "request_id": "req_4e90",
        },
        {
            "timestamp": "2026-02-03T09:02:03Z",
            "user": "mchen",
            "action": "export",
            "case_id": "CASE-2307",
            "model_version": "v1.7.9",
            "data_version": "2025.12.15",
            "status": "success",
            "channel": "UI",
            "ip": "10.4.11.32",
            "latency_ms": 452,
            "notes": "Exported PDF profile for briefing package.",
            "request_id": "req_4db8",
        },
        {
            "timestamp": "2026-02-02T23:44:17Z",
            "user": "npatel",
            "action": "login",
            "case_id": "N/A",
            "model_version": "v1.8.2",
            "data_version": "2026.01.30",
            "status": "success",
            "channel": "UI",
            "ip": "10.4.9.10",
            "latency_ms": 138,
            "notes": "Authenticated via SSO.",
            "request_id": "req_4c71",
        },
        {
            "timestamp": "2026-02-02T18:21:55Z",
            "user": "jlee",
            "action": "save",
            "case_id": "CASE-2311",
            "model_version": "v1.8.1",
            "data_version": "2026.01.22",
            "status": "success",
            "channel": "API",
            "ip": "10.4.22.19",
            "latency_ms": 305,
            "notes": "Submitted revised MO description via API client.",
            "request_id": "req_4c05",
        },
        {
            "timestamp": "2026-02-02T11:06:48Z",
            "user": "asoto",
            "action": "predict",
            "case_id": "CASE-2307",
            "model_version": "v1.8.1",
            "data_version": "2026.01.22",
            "status": "failed",
            "channel": "UI",
            "ip": "10.4.18.6",
            "latency_ms": 1260,
            "notes": "Model request timed out; retry recommended.",
            "request_id": "req_4bb2",
        },
        {
            "timestamp": "2026-02-01T20:14:09Z",
            "user": "admin",
            "action": "export",
            "case_id": "CASE-2299",
            "model_version": "v1.7.8",
            "data_version": "2025.12.01",
            "status": "success",
            "channel": "UI",
            "ip": "10.4.1.4",
            "latency_ms": 512,
            "notes": "Bulk export of CSV audit bundle.",
            "request_id": "req_4a10",
        },
    ]
    return from_storage + demo


def prepare_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def reset_filters(defaults: dict) -> None:
    st.session_state["audit_user_filter"] = defaults["user"]
    st.session_state["audit_action_filter"] = defaults["actions"]
    st.session_state["audit_case_filter"] = defaults["case"]
    st.session_state["audit_date_filter"] = defaults["dates"]
    st.session_state["audit_selected_label"] = None


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Audit Log",
        page_icon="AL",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    st.title("Audit Log")
    st.caption("Must-have for FR08 - trace user actions and model/data versions.")

    df = prepare_dataframe(seed_audit_log())
    default_date_range = (df["ts"].min().date(), df["ts"].max().date())
    default_filters = {
        "user": "All users",
        "actions": df["action"].unique().tolist(),
        "case": "",
        "dates": default_date_range,
    }

    if "audit_user_filter" not in st.session_state:
        reset_filters(default_filters)

    filters = st.columns([1.1, 1.1, 1.1, 1, 0.6])
    with filters[0]:
        users = ["All users"] + sorted(df["user"].unique().tolist())
        user_choice = st.selectbox("User", users, key="audit_user_filter")
    with filters[1]:
        action_choice = st.multiselect(
            "Action",
            options=["predict", "save", "export", "login"],
            default=st.session_state.get("audit_action_filter", default_filters["actions"]),
            key="audit_action_filter",
        )
    with filters[2]:
        date_input = st.date_input(
            "Date range",
            value=st.session_state.get("audit_date_filter", default_date_range),
            key="audit_date_filter",
        )
    with filters[3]:
        case_filter = st.text_input("Case ID", placeholder="CASE-2311", key="audit_case_filter")
    with filters[4]:
        st.button("Reset", on_click=reset_filters, args=(default_filters,), type="secondary")

    start_date, end_date = default_date_range
    if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        start_date, end_date = date_input
    elif isinstance(date_input, date):
        start_date = end_date = date_input

    filtered = df.copy()
    if user_choice != "All users":
        filtered = filtered[filtered["user"] == user_choice]
    if action_choice:
        filtered = filtered[filtered["action"].isin(action_choice)]
    if case_filter:
        filtered = filtered[filtered["case_id"].str.contains(case_filter.strip(), case=False)]
    filtered = filtered[
        (filtered["ts"].dt.date >= start_date) &
        (filtered["ts"].dt.date <= end_date)
    ]
    filtered = filtered.sort_values("ts", ascending=False)

    metrics = st.columns(4)
    metrics[0].metric("Events", len(filtered))
    metrics[1].metric("Unique users", filtered["user"].nunique())
    metrics[2].metric("Actions covered", ", ".join(sorted(filtered["action"].unique())))
    metrics[3].metric("Date span", f"{start_date} -> {end_date}")

    base_columns = ["timestamp", "user", "action", "case_id", "model_version", "data_version"]
    display_df = filtered[base_columns].rename(
        columns={
            "timestamp": "Timestamp (UTC)",
            "user": "User",
            "action": "Action",
            "case_id": "Case ID",
            "model_version": "Model ver",
            "data_version": "Data ver",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "Action": st.column_config.TextColumn(width="small"),
            "Case ID": st.column_config.TextColumn(width="small"),
            "Model ver": st.column_config.TextColumn(width="small"),
            "Data ver": st.column_config.TextColumn(width="small"),
        },
    )

    if not filtered.empty:
        labels = [f"{row['timestamp']} | {row['user']} | {row['action']} | {row['case_id']}" for _, row in filtered.iterrows()]
        current_label = st.session_state.get("audit_selected_label")
        default_index = labels.index(current_label) if current_label in labels else 0
        selected_label = st.selectbox("Select log entry", labels, index=default_index, key="audit_selected_label")
        idx = labels.index(selected_label) if selected_label and selected_label in labels else 0
        selected_row = filtered.iloc[idx]

        with st.expander("View details", expanded=False):
            st.markdown(f"**Timestamp (UTC):** {selected_row['timestamp']}")
            st.markdown(f"**User / Action:** {selected_row['user']} - {selected_row['action']}")
            st.markdown(f"**Case ID:** {selected_row['case_id']}")
            st.markdown(f"**Model ver / Data ver:** {selected_row['model_version']} / {selected_row['data_version']}")
            detail_cols = st.columns(3)
            detail_cols[0].markdown(f"**Status:** {selected_row['status']}")
            detail_cols[1].markdown(f"**Channel:** {selected_row['channel']}")
            detail_cols[2].markdown(f"**Latency:** {selected_row['latency_ms']} ms")
            st.markdown(f"**IP:** {selected_row['ip']}")
            st.markdown(f"**Request ID:** {selected_row['request_id']}")
            st.markdown("**Notes**")
            st.write(selected_row["notes"])

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered log (CSV)",
            data=csv_bytes,
            file_name="audit_log_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No audit events match the selected filters.")


if __name__ == "__main__":
    main()
