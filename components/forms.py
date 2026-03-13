from __future__ import annotations

from typing import Any, Dict, Tuple

import streamlit as st


def structured_case_form() -> Tuple[Dict[str, Any], bool]:
    """Render the structured intake form and return (payload, submitted)."""
    with st.form("structured_case_form"):
        st.markdown("#### Structured case details")
        with st.expander("Incident context", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                incident_type = st.text_input("Incident type", placeholder="e.g., warehouse burglary, stalking, fraud")
                location = st.text_input("Primary location", placeholder="City / district / site")
                time_of_day = st.selectbox(
                    "Time of day",
                    ["Unknown", "Daytime", "Evening", "Night"],
                    index=0,
                )
            with col2:
                incident_severity = st.slider(
                    "Perceived severity",
                    min_value=0,
                    max_value=10,
                    value=5,
                    help="Rough investigator assessment of harm / impact (0–10).",
                )
                prior_incidents = st.number_input(
                    "Known prior related incidents",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                )

        with st.expander("Victim / target profile", expanded=False):
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                victim_type = st.text_input("Victim / target type", placeholder="Person, small business, logistics depot…")
                age_group = st.selectbox(
                    "Primary victim age group",
                    ["Unknown", "Child", "Young adult", "Adult", "Older adult"],
                )
            with col_v2:
                vulnerability = st.selectbox(
                    "Perceived vulnerability",
                    ["Unknown", "Low", "Moderate", "High"],
                    index=1,
                )

        with st.expander("Modus operandi (MO)", expanded=False):
            entry_method = st.text_input("Entry / initial approach", placeholder="Rear window, phishing email, tailgating…")
            tools = st.text_input("Tools / methods observed", placeholder="Pry bar, vehicle, online platform…")
            repeat_pattern = st.selectbox(
                "Pattern of behaviour",
                ["Unknown", "Isolated", "Likely repeat", "Series / pattern confirmed"],
                index=0,
            )

        submitted = st.form_submit_button("Predict & save case", type="primary", use_container_width=True)

        payload: Dict[str, Any] = {
            "incident_type": incident_type,
            "location": location,
            "time_of_day": time_of_day,
            "incident_severity": incident_severity,
            "prior_incidents": prior_incidents,
            "victim_type": victim_type,
            "age_group": age_group,
            "vulnerability": vulnerability,
            "entry_method": entry_method,
            "tools": tools,
            "repeat_pattern": repeat_pattern,
        }
        return payload, submitted


def narrative_case_form() -> Tuple[str, bool]:
    """Render narrative-only input form and return (text, submitted)."""
    with st.form("narrative_case_form"):
        st.markdown("#### Narrative description")
        text = st.text_area(
            "Case narrative",
            placeholder=(
                "Summarise the incident using operational language: sequence of events, locations, victim context, "
                "and any observed patterns or escalation cues."
            ),
            height=220,
        )
        submitted = st.form_submit_button("Predict & save case", type="primary", use_container_width=True)
        return text, submitted


__all__ = ["structured_case_form", "narrative_case_form"]

