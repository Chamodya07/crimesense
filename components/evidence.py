from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_evidence_items(items: List[Dict[str, Any]]) -> None:
    st.markdown("### Similar cases / narrative snippets")
    if not items:
        st.info(
            "No similar cases were found in the local index yet. "
            "As you save more cases, this panel will surface relevant examples."
        )
        return

    for item in items:
        cid = item.get("case_id", "Unknown")
        score = item.get("similarity_score", 0.0)
        meta = item.get("metadata", {}) or {}
        label = f"{cid} · similarity {score:.2f}"
        with st.expander(label, expanded=False):
            st.markdown(f"**Case ID:** {cid}")
            st.markdown(f"**Similarity score:** {score:.3f}")
            if "created_at" in meta:
                st.markdown(f"**Created:** {meta['created_at']}")
            if "risk" in meta and meta["risk"]:
                st.markdown(f"**Stored risk category:** {meta['risk']}")
            st.markdown("**Snippet**")
            st.write(item.get("snippet", ""))


__all__ = ["render_evidence_items"]

