"""
Chat message rendering, including the Trust Panel — the single biggest UX
differentiator versus a generic chatbot (spec section 21).
"""
from __future__ import annotations

import streamlit as st

from app.llm.schemas import ChatResponse
from app.ui.components.evidence_panel import render_evidence_panel
from app.ui.components.objection_card import render_objection_response
from app.ui.components.scheme_card import render_scheme_card


def render_trust_panel(response: ChatResponse) -> None:
    tc = response.trust_check
    check = lambda ok: "✅" if ok else "⚠️"

    st.markdown(
        f"""
        <div class="km-trust-panel">
          <div class="km-trust-title">🛡 TRUST CHECK</div>
          <div class="km-trust-row"><span>Evidence found</span><span>{check(tc.evidence_found)}</span></div>
          <div class="km-trust-row"><span>Source verified</span><span>{check(tc.source_verified)}</span></div>
          <div class="km-trust-row"><span>Hallucination check</span><span>{check(tc.hallucination_checked)} ({tc.claims_removed} claim(s) removed)</span></div>
          <div class="km-trust-row"><span>Eligibility complete</span><span>{check(tc.eligibility_complete)}</span></div>
          <div class="km-trust-row"><span><b>Confidence</b></span><span><b>{tc.confidence.upper()}</b></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_refusal(response: ChatResponse) -> None:
    st.markdown(f'<div class="km-refusal-card">🚫 {response.answer}</div>', unsafe_allow_html=True)
    if response.guardrail_triggered:
        st.caption(f"Guardrail: `{response.guardrail_triggered}`")


def render_chat_response(response: ChatResponse, is_objection: bool = False) -> None:
    if response.refused:
        render_refusal(response)
        render_trust_panel(response)
        return

    if is_objection:
        render_objection_response(response)
    else:
        st.markdown('<div class="km-card">', unsafe_allow_html=True)
        st.write(response.answer)
        st.markdown("</div>", unsafe_allow_html=True)

    if response.schemes:
        st.markdown("#### 🌾 Scheme matches")
        evidence_by_id = {c.chunk_id: c for c in response.evidence}
        for i, scheme in enumerate(response.schemes):
            render_scheme_card(scheme, evidence_by_id, index=i)

    if response.missing_information:
        st.markdown("**What is still missing to complete this picture:**")
        for item in response.missing_information:
            st.markdown(f"- {item}")

    if not response.schemes and not is_objection:
        render_evidence_panel(response.evidence, key_prefix="general")

    render_trust_panel(response)

    if response.follow_up_questions:
        st.markdown("**To narrow this down, I need a couple of details:**")
        for q in response.follow_up_questions:
            st.markdown(f"- {q}")

    if response.debug:
        with st.expander("🛠 Debug: retrieval & pipeline internals", expanded=False):
            st.json(response.debug)
