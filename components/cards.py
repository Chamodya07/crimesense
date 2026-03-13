from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Mapping, Sequence

import pandas as pd
import streamlit as st

from config import MODELS
from services.storage_service import CaseRecord


def _risk_from_case(case: CaseRecord) -> str:
    fusion = case.get("fusion_output") or {}
    return str(fusion.get("final_risk_category") or "").title() or "Unknown"


def render_kpis(cases: Sequence[CaseRecord]) -> None:
    total_cases = len(cases)
    last_time = max((c.get("created_at") for c in cases if c.get("created_at")), default="—")

    risks = [_risk_from_case(c) for c in cases if c.get("fusion_output")]
    avg_risk_score = None
    if cases:
        scores = []
        for c in cases:
            fusion = c.get("fusion_output") or {}
            score = fusion.get("final_risk_score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
        if scores:
            avg_risk_score = sum(scores) / len(scores)

    cols = st.columns(4)
    cols[0].metric("Cases predicted", f"{total_cases}")
    cols[1].metric("Last prediction time", last_time)
    cols[2].metric(
        "Avg risk score",
        f"{avg_risk_score:.2f}" if avg_risk_score is not None else "—",
    )
    cols[3].metric(
        "Model versions",
        f"T: {MODELS.tabular_model_version} · N: {MODELS.nlp_model_version}",
    )


def render_risk_distribution(cases: Sequence[CaseRecord]) -> None:
    risks = [_risk_from_case(c) for c in cases if c.get("fusion_output")]
    st.markdown("**Risk distribution (saved cases)**")
    if not risks:
        st.info("No saved predictions yet. Run a new case to populate risk distribution.")
        return
    counts = Counter(risks)
    df = pd.DataFrame({"Risk": list(counts.keys()), "Count": list(counts.values())})
    st.bar_chart(df.set_index("Risk"))


def render_recent_cases_table(cases: Iterable[CaseRecord], max_rows: int = 10) -> None:
    st.markdown("**Recent cases**")
    items: List[Mapping] = sorted(
        cases,
        key=lambda c: c.get("created_at") or "",
        reverse=True,
    )[:max_rows]
    if not items:
        st.info("No cases stored yet.")
        return

    rows = []
    for c in items:
        fusion = c.get("fusion_output") or {}
        final = fusion.get("final", {})
        motive_obj = final.get("motive", {})
        motive = motive_obj.get("pred") if isinstance(motive_obj, dict) else motive_obj
        if not motive:
            motive = fusion.get("explanations", {}).get("tabular", {}).get("motive")
        rows.append(
            {
                "Case ID": c.get("case_id"),
                "Created": c.get("created_at"),
                "Risk": fusion.get("final_risk_category"),
                "Score": fusion.get("final_risk_score"),
                "Motive": motive,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Select a case from History to open its full profile.")


__all__ = ["render_kpis", "render_risk_distribution", "render_recent_cases_table"]

