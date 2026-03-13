from __future__ import annotations

import sys
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


def summary_row() -> None:
    profile_count = "8K"
    try:
        from services.storage_service import load_cases
        n = len(load_cases())
        if n > 0:
            profile_count = str(n)
    except Exception:
        pass
    cols = st.columns(4)
    cols[0].metric("Crimes Reported for 2025", "11.8M", "+2.5%")
    cols[1].metric("Crimes Reported this Week", "8.236K", "-1.2%")
    cols[2].metric("Totals (Sri Lanka)", "2.352M", "+11%")
    cols[3].metric("Available Criminal Profiles", profile_count, "+5.2%")


def gauge_block() -> None:
    st.markdown("**Target Crime Solving Rate by 2025**")
    st.progress(0.67)
    st.caption("67% achieved - Remaining 33%")


def radar_block() -> None:
    st.markdown("**Most Common Crime Types**")
    data = pd.DataFrame(
        {
            "Crime": ["Burglary", "Fraud", "Assault", "Auto theft", "Cyber"],
            "Count": [42, 35, 28, 24, 31],
        }
    )
    st.bar_chart(data.set_index("Crime"))


def country_table() -> None:
    st.markdown("**Crime Rate by Country**")
    data = pd.DataFrame(
        {
            "Country": ["United States", "Australia", "China", "Germany", "Romania", "Japan", "Netherlands"],
            "Rate": ["27.5%", "11.2%", "9.4%", "8%", "7.9%", "6.1%", "5.9%"],
            "Cases": ["4.5M", "2.3M", "2M", "1.7M", "1.6M", "1.2M", "1M"],
        }
    )
    st.dataframe(data, hide_index=True, use_container_width=True)


def spend_block() -> None:
    st.markdown("**Spend on Investigations by Country**")
    spend = pd.DataFrame({"Country": ["User Name"] * 5, "Amount": [1.2, 0.8, 0.645, 0.59, 0.342]})
    st.bar_chart(spend.set_index("Country"))


def quick_links() -> None:
    st.markdown("### Quick links")
    st.markdown("- **Welcome** (home)\n- **Profile Intake**\n- **Past Predictions**\n- **Audit Log**\n- **Dashboard**")


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Dashboard",
        page_icon="DB",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    st.title("Dashboard")
    tabs = st.tabs(["Overview", "Audit", "Cases", "Datasets", "Reports"])
    with tabs[0]:
        summary_row()

        top = st.columns((1.4, 1, 1.2))
        with top[0]:
            st.container(border=True, height=260)
            with st.container(border=True):
                gauge_block()
        with top[1]:
            st.container(border=True, height=260)
            radar_block()
        with top[2]:
            st.container(border=True, height=260)
            st.markdown("**Crime Frequency by Location**")
            st.caption("Placeholder map - connect to geo service for heatmap.")

        bottom = st.columns((1.2, 1, 1))
        with bottom[0]:
            st.container(border=True, height=320)
            country_table()
        with bottom[1]:
            st.container(border=True, height=320)
            spend_block()
        with bottom[2]:
            st.container(border=True, height=320)
            quick_links()
    for tab in tabs[1:]:
        with tab:
            st.info("Coming soon - hook to your backend data.")


if __name__ == "__main__":
    main()
