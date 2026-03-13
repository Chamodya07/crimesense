"""Explanation and fusion display components."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_tabular_explanations(expl: Dict[str, Any], tabular_top_features: List[Dict[str, Any]] | None = None) -> None:
    st.markdown("#### Structured signal (tabular model)")
    cols = st.columns(3)
    cols[0].metric("Risk (tabular)", expl.get("risk_category", "—"), delta=None)
    cols[1].metric("Score", f"{expl.get('risk_score', '—')}")
    cols[2].markdown(f"**Motive (tabular):** {expl.get('motive', '—')}")

    if tabular_top_features:
        st.markdown("**Top features (SHAP)**")
        for item in tabular_top_features[:10]:
            f = item.get("feature", "")
            v = item.get("value", 0)
            st.caption(f"`{f}`: {v:.4f}")
    else:
        st.caption(
            "Feature-level explanations (e.g., SHAP values) can be added here. "
            "For the prototype, this panel shows summary signals only."
        )


def render_narrative_explanations(expl: Dict[str, Any]) -> None:
    st.markdown("#### Narrative signal (text model)")
    topk = expl.get("topk", [])
    if topk:
        st.markdown("**Motive top-k from narrative**")
        for i, item in enumerate(topk[:5], 1):
            lbl = item.get("label", "—")
            prob = item.get("prob", 0)
            st.markdown(f"{i}. **{lbl}** ({prob:.2%})")
    else:
        cols = st.columns(3)
        cols[0].metric("Risk (narrative)", expl.get("risk_category", "—"))
        cols[1].metric("Score", f"{expl.get('risk_score', '—')}")
        cols[2].markdown(f"**Experience (NLP):** {expl.get('experience_level', '—')}")
        key_phrases: List[str] = list(expl.get("key_phrases", []))
        if key_phrases:
            st.markdown("**Key phrases / cues extracted from narrative**")
            st.write(", ".join(f"`{p}`" for p in key_phrases))
        else:
            st.caption("No narrative provided — NLP motive prediction skipped.")


def render_fused_summary(
    final_risk: str,
    risk_score: float | str | None = None,
    offender_type: str | None = None,
    motive_pred: str | None = None,
    motive_confidence: float | None = None,
    motive_band: str | None = None,
    fusion_version: str | None = None,
) -> None:
    st.markdown("### Fused offender profile (combined models)")
    cols = st.columns(4)
    # risk level
    cols[0].metric("Risk level", final_risk)
    # optional numeric score
    cols[1].metric("Risk score", f"{risk_score}" if risk_score not in (None, "") else "—")
    # offender type
    cols[2].metric("Offender type", offender_type or "—")
    # final motive with confidence/band as delta text
    if motive_pred:
        delta_text = None
        if motive_confidence is not None or motive_band:
            conf_str = f"{motive_confidence:.2f}" if motive_confidence is not None else "—"
            delta_text = f"{conf_str}{' (' + motive_band + ')' if motive_band else ''}"
        cols[3].metric("Final motive", motive_pred, delta=delta_text)
    else:
        cols[3].metric("Final motive", "—")
    if fusion_version:
        st.caption(f"Fusion model version: `{fusion_version}`")


def render_fusion_meta(fusion_meta: Dict[str, Any] | None) -> None:
    if not fusion_meta:
        return
    warnings = fusion_meta.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w, icon="⚠️")


def render_ethical_notice() -> None:
    st.warning(
        "This tool is **decision support only**. Predictions are generated from historical patterns and may reflect "
        "biases in the data. Outputs **must not** be used as sole evidence and always require review by a trained "
        "investigator or analyst.",
        icon="⚖️",
    )


__all__ = [
    "render_tabular_explanations",
    "render_narrative_explanations",
    "render_fused_summary",
    "render_fusion_meta",
    "render_ethical_notice",
]
