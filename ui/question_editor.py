"""
Question Editor — Side-by-side comparison of refined question versions.

Displays Easy/Medium/Hard versions with their XGBoost validation results.
"""
import streamlit as st


def render_question_card(level: str, content: str, validation: dict):
    """Render a single question version card."""
    colors = {
        "easy": ("#10b981", "rgba(16, 185, 129, 0.1)", "🟢"),
        "medium": ("#f59e0b", "rgba(245, 158, 11, 0.1)", "🟡"),
        "hard": ("#ef4444", "rgba(239, 68, 68, 0.1)", "🔴")
    }
    
    color, bg_color, icon = colors.get(level.lower(), ("#6b7280", "rgba(107,114,128,0.1)", "⚪"))
    
    # Validation status
    if validation:
        predicted = validation.get("predicted_difficulty", "?")
        matches = validation.get("matches", False)
        di = validation.get("difficulty_index", 0)
        confidence = validation.get("confidence", 0)
        
        match_badge = "✅ Validated" if matches else f"⚠️ Predicted: {predicted}"
        stats_line = f"DI: {di:.2f} | Confidence: {confidence:.1%}"
    else:
        match_badge = "—"
        stats_line = ""
    
    st.markdown(f"""
<div style="
    border: 2px solid {color}; 
    border-radius: 12px; 
    padding: 16px; 
    background: {bg_color};
    margin-bottom: 8px;
">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:18px; font-weight:700; color:{color};">
            {icon} {level.upper()}
        </span>
        <span style="font-size:12px; padding:2px 8px; border-radius:8px; 
              background:rgba(0,0,0,0.2); color:#e2e8f0;">
            {match_badge}
        </span>
    </div>
    <div style="font-size:11px; color:#94a3b8; margin-bottom:8px;">{stats_line}</div>
</div>
""", unsafe_allow_html=True)
    
    if content:
        st.markdown(content)
    else:
        st.caption("No version generated for this level.")


def render_question_editor(result: dict):
    """Render the side-by-side question comparison view."""
    refined = result.get("refined_questions", {})
    validation = result.get("validation_results", {})
    justification = result.get("difficulty_justification", "")
    
    if not any(refined.values()):
        st.info("No refined questions available yet. Run the analysis first.")
        return
    
    st.subheader("📝 Refined Question Versions")
    
    # Three-column layout
    cols = st.columns(3)
    
    for col, level in zip(cols, ["easy", "medium", "hard"]):
        with col:
            render_question_card(
                level=level,
                content=refined.get(level, ""),
                validation=validation.get(level, {})
            )
    
    # Justification report
    if justification:
        with st.expander("📊 Difficulty Justification", expanded=True):
            st.markdown(justification)


def render_original_vs_refined(original: str, refined: dict, validation: dict):
    """Render original question alongside the refined version matching the target."""
    st.subheader("🔄 Before & After")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Question**")
        st.code(original, language=None)
    
    with col2:
        # Show the medium version by default, or the target
        best_version = refined.get("medium") or refined.get("easy") or refined.get("hard", "")
        st.markdown("**Refined Question**")
        if best_version:
            st.markdown(best_version)
        else:
            st.caption("No refined version available.")
