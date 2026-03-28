from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
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
EXCLUDED_ACTIONS = {"login", "logout"}
ACTION_OPTIONS = ["predict", "save", "export"]
DEFAULT_LOOKBACK_DAYS = 7


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


def _event_timestamp(record: dict[str, Any]) -> datetime | None:
    return _parse_utc_timestamp(record.get("ts_local") or record.get("ts_utc"))


def _format_timestamp(value: Any) -> str:
    parsed = _parse_utc_timestamp(value)
    if parsed is None:
        return "-"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "-", "N/A", "None"}
    return str(value).strip() in {"", "-", "N/A", "None"}


def _first_present(*values: Any) -> Any:
    for value in values:
        if not is_empty(value):
            return value
    return ""


def _prepare_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    fallback_data_ver = "N/A"
    try:
        from services.audit_service import compute_data_ver

        fallback_data_ver = compute_data_ver() or "N/A"
    except Exception:
        fallback_data_ver = "N/A"

    prepared = []
    for record in records:
        ts_dt = _event_timestamp(record)
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        prepared.append(
            {
                "Timestamp (UTC)": _format_timestamp(ts_dt),
                "User": str(record.get("user") or "-"),
                "Action": str(record.get("action") or "-"),
                "Case ID": str(record.get("case_id") or "N/A"),
                "_page": str(record.get("page") or "-"),
                "_meta": meta,
                "_status": _first_present(record.get("status"), meta.get("status")),
                "_channel": _first_present(record.get("channel"), record.get("page"), meta.get("channel")),
                "_latency": _first_present(
                    record.get("latency_ms"),
                    record.get("latency"),
                    meta.get("latency_ms"),
                    meta.get("latency"),
                ),
                "_error": _first_present(record.get("error"), meta.get("error")),
                "_message": _first_present(record.get("message"), meta.get("message")),
                "_ts_dt": ts_dt,
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
                "_page",
                "_meta",
                "_status",
                "_channel",
                "_latency",
                "_error",
                "_message",
                "_ts_dt",
            ]
        )
    return df


def _default_date_range() -> tuple[date, date]:
    today = date.today()
    return today - timedelta(days=DEFAULT_LOOKBACK_DAYS), today


def _reset_filters() -> None:
    st.session_state["audit_user_filter"] = "All users"
    st.session_state["audit_action_filter"] = ACTION_OPTIONS.copy()
    st.session_state["audit_date_range"] = _default_date_range()
    st.session_state["audit_case_id_filter"] = ""
    st.session_state["audit_search_filter"] = ""
    st.session_state["audit_selected_label"] = None
    st.rerun()


def _ensure_filter_state(users: list[str]) -> None:
    if "audit_user_filter" not in st.session_state:
        st.session_state["audit_user_filter"] = "All users"
    elif st.session_state["audit_user_filter"] not in users:
        st.session_state["audit_user_filter"] = "All users"

    if "audit_action_filter" not in st.session_state:
        st.session_state["audit_action_filter"] = ACTION_OPTIONS.copy()
    elif not isinstance(st.session_state["audit_action_filter"], list):
        st.session_state["audit_action_filter"] = ACTION_OPTIONS.copy()
    else:
        cleaned_actions = [
            action for action in st.session_state["audit_action_filter"] if action in ACTION_OPTIONS
        ]
        if st.session_state["audit_action_filter"] and not cleaned_actions:
            st.session_state["audit_action_filter"] = ACTION_OPTIONS.copy()
        else:
            st.session_state["audit_action_filter"] = cleaned_actions

    if "audit_date_range" not in st.session_state:
        st.session_state["audit_date_range"] = _default_date_range()
    elif not isinstance(st.session_state["audit_date_range"], (list, tuple)) or len(st.session_state["audit_date_range"]) != 2:
        st.session_state["audit_date_range"] = _default_date_range()

    if "audit_case_id_filter" not in st.session_state:
        st.session_state["audit_case_id_filter"] = ""

    if "audit_search_filter" not in st.session_state:
        st.session_state["audit_search_filter"] = ""

    if "audit_selected_label" not in st.session_state:
        st.session_state["audit_selected_label"] = None


def load_audit_events(limit: int = 500) -> list[dict[str, Any]]:
    try:
        from firebase_admin import firestore

        from services.firebase_service import FirebaseConfigError, init_firebase
    except Exception as err:
        st.info(f"Firebase not configured. {err}")
        return []

    try:
        db = init_firebase()
        docs = (
            db.collection("audit_events")
            .order_by("ts_utc", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        events = [{"id": doc.id, **(doc.to_dict() or {})} for doc in docs]
        events = [e for e in events if e.get("action") not in EXCLUDED_ACTIONS]
        return events
    except FirebaseConfigError as err:
        st.info(f"Firebase not configured. {err}")
        return []
    except Exception as err:
        st.info(f"Unable to load audit events right now. {err}")
        return []


def _filter_events(
    records: list[dict[str, Any]],
    user_filter: str,
    action_filter: list[str],
    case_filter: str,
    date_range: Any,
    search_filter: str,
) -> list[dict[str, Any]]:
    normalized_user = str(user_filter or "").strip()
    normalized_actions = {str(item).strip().lower() for item in action_filter if str(item).strip()}
    normalized_case = str(case_filter or "").strip().lower()
    normalized_search = str(search_filter or "").strip().lower()

    start_date, end_date = _default_date_range()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, date):
        start_date = end_date = date_range

    filtered: list[dict[str, Any]] = []
    for record in records:
        event_user = str(record.get("user") or "").strip()
        event_action = str(record.get("action") or "").strip().lower()
        event_case = str(record.get("case_id") or "").strip()
        event_ts = _event_timestamp(record)
        event_date = event_ts.date() if event_ts is not None else None

        if normalized_user and normalized_user != "All users" and event_user != normalized_user:
            continue
        if normalized_actions and event_action not in normalized_actions:
            continue
        if normalized_case and normalized_case not in event_case.lower():
            continue
        if event_date is None or event_date < start_date or event_date > end_date:
            continue
        if normalized_search:
            haystack = " ".join([event_user, event_action, event_case]).lower()
            if normalized_search not in haystack:
                continue

        filtered.append(record)

    filtered.sort(key=lambda record: _event_timestamp(record) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return filtered


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
        import services.audit_service as audit_service
    except Exception as err:
        st.error(f"Audit service unavailable: {err}")
        return

    current_user_getter = getattr(audit_service, "get_current_user", None)
    if not callable(current_user_getter):
        current_user_getter = getattr(audit_service, "get_current_user_label", None)
    current_user_label = current_user_getter() if callable(current_user_getter) else "unknown"
    st.caption(f"Current user resolved as: {current_user_label}")

    all_events = load_audit_events(limit=500)
    st.caption(f"First event keys: {list(all_events[0].keys()) if all_events else []}")
    users = ["All users"] + sorted(
        {str(item.get("user") or "").strip() for item in all_events if str(item.get("user") or "").strip()}
    )
    _ensure_filter_state(users)

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
        st.button("Reset", on_click=_reset_filters, type="secondary")

    filtered_events = _filter_events(
        records=all_events,
        user_filter=user_choice,
        action_filter=action_choice,
        case_filter=case_filter,
        date_range=date_input,
        search_filter=st.session_state.get("audit_search_filter", ""),
    )
    filtered = _prepare_dataframe(filtered_events)

    if not filtered.empty and filtered["_ts_dt"].notna().any():
        filtered_ts = filtered["_ts_dt"].dropna()
        min_date = filtered_ts.min().date()
        max_date = filtered_ts.max().date()
        date_span = f"{min_date} -> {max_date}"
    else:
        date_span = "-"

    metrics = st.columns(4)
    metrics[0].metric("Events", len(filtered))
    metrics[1].metric("Unique users", filtered["User"].nunique() if not filtered.empty else 0)
    metrics[2].metric("Actions covered", ", ".join(sorted(filtered["Action"].unique())) if not filtered.empty else "-")
    metrics[3].metric("Date span", date_span)

    display_columns = ["Timestamp (UTC)", "User", "Action", "Case ID"]
    display_df = filtered[display_columns] if not filtered.empty else pd.DataFrame(columns=display_columns)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp (UTC)": st.column_config.TextColumn(width="medium"),
            "Action": st.column_config.TextColumn(width="small"),
            "Case ID": st.column_config.TextColumn(width="small"),
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
            detail_pairs = [
                ("Timestamp (UTC)", selected_row.get("Timestamp (UTC)")),
                ("User", selected_row.get("User")),
                ("Action", selected_row.get("Action")),
                ("Case ID", selected_row.get("Case ID")),
                ("Status", selected_row.get("_status")),
                ("Channel", selected_row.get("_channel")),
            ]
            summary_parts = [str(value).strip() for _, value in detail_pairs if not is_empty(value)]
            if summary_parts:
                st.caption(" | ".join(summary_parts))

            for label, value in detail_pairs:
                if is_empty(value):
                    continue
                st.markdown(f"**{label}:** {value}")

            latency_value = selected_row.get("_latency")
            if not is_empty(latency_value):
                latency_text = str(latency_value).strip()
                if not latency_text.lower().endswith("ms"):
                    latency_text = f"{latency_text} ms"
                st.markdown(f"**Latency:** {latency_text}")

            status_value = str(selected_row.get("_status") or "").strip().lower()
            error_value = selected_row.get("_error")
            message_value = selected_row.get("_message")
            if not is_empty(error_value):
                st.markdown(f"**Error:** {error_value}")
            if not is_empty(message_value) and (status_value != "success" or not is_empty(error_value)):
                st.markdown(f"**Message:** {message_value}")

            meta_value = selected_row["_meta"] if isinstance(selected_row["_meta"], dict) else {}
            if meta_value:
                with st.expander("Metadata", expanded=False):
                    st.json(meta_value)

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
