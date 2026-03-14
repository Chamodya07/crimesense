from __future__ import annotations

import streamlit as st


PAGES = [
    {
        "key": "dashboard",
        "title": "Dashboard",
        "path": "pages/dashboard.py",
        "description": "Live case activity, trends, and operational summaries.",
    },
    {
        "key": "profile",
        "title": "Profile Intake",
        "path": "pages/profile.py",
        "description": "Create a new prediction using structured case details.",
    },
    {
        "key": "history",
        "title": "Past Predictions",
        "path": "pages/history.py",
        "description": "Review saved cases, inspect model outputs, and export prior profiles for analysis.",
    },
    {
        "key": "audit",
        "title": "Audit Log",
        "path": "pages/audit_log.py",
        "description": "Inspect usage events, timestamps, and model traceability.",
    },
]


def render_page_cards(current_page: str, title: str = "Pages") -> None:
    items = [item for item in PAGES if item["key"] != current_page]
    if not items:
        return

    st.markdown(f"### {title}")
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            with st.container(border=True, height=180):
                st.markdown(f"**{item['title']}**")
                st.caption(item["description"])
                if hasattr(st, "page_link"):
                    st.page_link(item["path"], label=f"Open {item['title']}", use_container_width=True)
                else:
                    if st.button(
                        f"Open {item['title']}",
                        key=f"nav_{current_page}_{item['key']}",
                        use_container_width=True,
                    ):
                        st.switch_page(item["path"])


__all__ = ["render_page_cards"]
