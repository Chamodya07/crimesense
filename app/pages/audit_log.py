from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.auth_service import render_auth_status

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"
ACTION_OPTIONS = ["predict", "save", "export", "login", "logout"]


def inject_styles() -> None:
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _parse_utc_timestamp(value: Any) -> datetime | None:
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


def _format_timestamp(value: Any) -> str:
    parsed = _parse_utc_timestamp(value)
    if parsed is None:
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _prepare_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    prepared = []
    for record in records:
        ts_utc = record.get("ts_utc") or record.get("ts_local")
        prepared.append(
            {
                "Timestamp (UTC)": _format_timestamp(ts_utc),
                "User": str(record.get("user") or "-"),
                "Action": str(record.get("action") or "-"),
                "Case ID": str(record.get("case_id") or "-"),
                "Model ver": str(record.get("model_ver") or "-"),
                "Data ver": str(record.get("data_ver") or "-"),
                "_page": str(record.get("page") or "-"),
                "_meta": record.get("meta") or {},
                "_ts_dt": _parse_utc_timestamp(ts_utc),
            }
        )

    df = pd.DataFrame(prepared)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "Timestamp (UTC)",
                "User",
                "Action",
                "Case ID",
                "Model ver",
                "Data ver",
                "_page",
                "_meta",
                "_ts_dt",
            ]
        )
    return df


def _reset_filters(default_dates: tuple[date, date]) -> None:
    st.session_state["audit_user_filter"] = "All users"
    st.session_state["audit_action_filter"] = ACTION_OPTIONS.copy()
    st.session_state["audit_date_range"] = default_dates
    st.session_state["audit_case_id_filter"] = ""
    st.session_state["audit_selected_label"] = None
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Audit Log",
        page_icon="AL",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    render_auth_status("audit")

    st.title("Audit Log")
    st.caption("Must-have for FR08 - trace user actions and model/data versions.")

    try:
        from services.audit_service import list_distinct_users, query_events
    except Exception as err:
        st.error(f"Audit service unavailable: {err}")
        return

    all_events = query_events(limit=500)
    all_df = _prepare_dataframe(all_events)

    if not all_df.empty and all_df["_ts_dt"].notna().any():
        ts_series = all_df["_ts_dt"].dropna()
        default_date_range = (ts_series.min().date(), ts_series.max().date())
    else:
        today = date.today()
        default_date_range = (today, today)

    users = ["All users"] + list_distinct_users(limit=200)

    if "audit_user_filter" not in st.session_state:
        st.session_state["audit_user_filter"] = "All users"
    if "audit_action_filter" not in st.session_state:
        st.session_state["audit_action_filter"] = ACTION_OPTIONS.copy()
    if "audit_date_range" not in st.session_state:
        st.session_state["audit_date_range"] = default_date_range
    if "audit_case_id_filter" not in st.session_state:
        st.session_state["audit_case_id_filter"] = ""
    if "audit_selected_label" not in st.session_state:
        st.session_state["audit_selected_label"] = None

    if st.session_state["audit_user_filter"] not in users:
        st.session_state["audit_user_filter"] = "All users"
    if not isinstance(st.session_state["audit_date_range"], (list, tuple)) or len(st.session_state["audit_date_range"]) != 2:
        st.session_state["audit_date_range"] = default_date_range
    st.session_state["audit_action_filter"] = [
        action for action in st.session_state["audit_action_filter"] if action in ACTION_OPTIONS
    ]

    filters = st.columns([1.1, 1.1, 1.1, 1, 0.6])
    with filters[0]:
        user_choice = st.selectbox("User", users, key="audit_user_filter")
    with filters[1]:
        action_choice = st.multiselect("Action", options=ACTION_OPTIONS, key="audit_action_filter")
    with filters[2]:
        date_input = st.date_input("Date range", key="audit_date_range")
    with filters[3]:
        case_filter = st.text_input("Case ID", placeholder="CASE-2311", key="audit_case_id_filter")
    with filters[4]:
        st.button("Reset", on_click=_reset_filters, args=(default_date_range,), type="secondary")

    start_date, end_date = default_date_range
    if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        start_date, end_date = date_input
    elif isinstance(date_input, date):
        start_date = end_date = date_input

    filtered_events = query_events(
        user=None if user_choice == "All users" else user_choice,
        actions=action_choice or None,
        case_id=case_filter,
        start_date=start_date,
        end_date=end_date,
        limit=500,
    )
    filtered = _prepare_dataframe(filtered_events)

    metrics = st.columns(4)
    metrics[0].metric("Events", len(filtered))
    metrics[1].metric("Unique users", filtered["User"].nunique() if not filtered.empty else 0)
    metrics[2].metric("Actions covered", ", ".join(sorted(filtered["Action"].unique())) if not filtered.empty else "-")
    metrics[3].metric("Date span", f"{start_date} -> {end_date}")

    display_columns = ["Timestamp (UTC)", "User", "Action", "Case ID", "Model ver", "Data ver"]
    display_df = filtered[display_columns] if not filtered.empty else pd.DataFrame(columns=display_columns)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp (UTC)": st.column_config.TextColumn(width="medium"),
            "Action": st.column_config.TextColumn(width="small"),
            "Case ID": st.column_config.TextColumn(width="small"),
            "Model ver": st.column_config.TextColumn(width="small"),
            "Data ver": st.column_config.TextColumn(width="small"),
        },
    )

    if not filtered.empty:
        labels = [
            f"{row['Timestamp (UTC)']} | {row['User']} | {row['Action']} | {row['Case ID']}"
            for _, row in filtered.iterrows()
        ]
        if st.session_state["audit_selected_label"] not in labels:
            st.session_state["audit_selected_label"] = labels[0]

        selected_label = st.selectbox("Select log entry", labels, key="audit_selected_label")
        selected_row = filtered.iloc[labels.index(selected_label)]

        with st.expander("View details", expanded=False):
            st.markdown(f"**Timestamp (UTC):** {selected_row['Timestamp (UTC)']}")
            st.markdown(f"**User / Action:** {selected_row['User']} - {selected_row['Action']}")
            st.markdown(f"**Case ID:** {selected_row['Case ID']}")
            st.markdown(f"**Model ver / Data ver:** {selected_row['Model ver']} / {selected_row['Data ver']}")
            detail_cols = st.columns(3)
            detail_cols[0].markdown("**Status:** success")
            detail_cols[1].markdown(f"**Channel:** {selected_row['_page']}")
            detail_cols[2].markdown("**Latency:** -")
            st.markdown("**IP:** -")
            st.markdown("**Request ID:** -")
            st.markdown("**Notes**")
            meta_value = selected_row["_meta"] if isinstance(selected_row["_meta"], dict) else {}
            if meta_value:
                st.write(meta_value)
            else:
                st.write("No additional metadata.")

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
