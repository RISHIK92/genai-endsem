"""
Analysis Dashboard — ML statistics visualization.

Renders XGBoost prediction results, SHAP-style feature importance,
and audit warnings in a visual dashboard.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from config.settings import DIFFICULTY_CLASSES


def render_difficulty_gauge(ml_stats: dict):
    """Render a difficulty gauge meter."""
    difficulty = ml_stats.get("difficulty", "Unknown")
    difficulty_index = ml_stats.get("difficulty_index", 0.5)
    confidence = ml_stats.get("confidence", 0.0)
    
    # Color mapping
    colors = {"Easy": "#10b981", "Medium": "#f59e0b", "Hard": "#ef4444"}
    color = colors.get(difficulty, "#6b7280")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=difficulty_index * 100,
        title={"text": f"Difficulty: {difficulty}", "font": {"size": 18, "color": "#e2e8f0"}},
        number={"suffix": "%", "font": {"color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 33], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [33, 66], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [66, 100], "color": "rgba(239, 68, 68, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "#f8fafc", "width": 3},
                "thickness": 0.8,
                "value": difficulty_index * 100
            }
        }
    ))
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Confidence: {confidence:.1%}")


def render_discrimination_bar(ml_stats: dict):
    """Render discrimination indicator."""
    discrimination = ml_stats.get("discrimination", "Unknown")
    disc_index = ml_stats.get("discrimination_index", 0.3)
    
    colors = {
        "Excellent": "#10b981",
        "Good": "#22c55e",
        "Fair": "#f59e0b",
        "Poor": "#ef4444"
    }
    color = colors.get(discrimination, "#6b7280")
    
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=disc_index,
        title={"text": f"Discrimination: {discrimination}", "font": {"size": 16, "color": "#e2e8f0"}},
        number={"font": {"color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#64748b"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 0.2], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [0.2, 0.3], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [0.3, 0.4], "color": "rgba(34, 197, 94, 0.2)"},
                {"range": [0.4, 1.0], "color": "rgba(16, 185, 129, 0.2)"}
            ]
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(ml_stats: dict):
    """Render SHAP-style feature importance chart."""
    importance = ml_stats.get("feature_importance", {})
    
    if not importance:
        st.info("Feature importance data not available.")
        return
    
    # Take top 8 features
    top_features = dict(list(importance.items())[:8])
    
    names = list(reversed(list(top_features.keys())))
    values = list(reversed(list(top_features.values())))
    
    # Color gradient based on importance
    colors = [f"rgba(99, 102, 241, {0.4 + 0.6 * v / (max(values) or 1)})" for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont={"color": "#e2e8f0"}
    ))
    
    fig.update_layout(
        title={"text": "Feature Importance", "font": {"size": 16, "color": "#e2e8f0"}},
        height=max(250, len(names) * 35 + 80),
        margin=dict(l=10, r=40, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "gridcolor": "rgba(100,116,139,0.2)",
            "tickfont": {"color": "#94a3b8"},
            "title": {"text": "Relative Importance", "font": {"color": "#94a3b8"}}
        },
        yaxis={
            "tickfont": {"color": "#e2e8f0"},
        },
        font={"color": "#e2e8f0"}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_probability_chart(ml_stats: dict):
    """Render difficulty class probabilities."""
    probs = ml_stats.get("difficulty_probabilities", {})
    
    if not probs:
        return
    
    colors = {"Easy": "#10b981", "Medium": "#f59e0b", "Hard": "#ef4444"}
    
    fig = go.Figure(go.Bar(
        x=list(probs.keys()),
        y=list(probs.values()),
        marker_color=[colors.get(k, "#6b7280") for k in probs.keys()],
        text=[f"{v:.1%}" for v in probs.values()],
        textposition="auto",
        textfont={"color": "#f8fafc", "size": 14}
    ))
    
    fig.update_layout(
        title={"text": "Difficulty Class Probabilities", "font": {"size": 14, "color": "#e2e8f0"}},
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={
            "gridcolor": "rgba(100,116,139,0.2)",
            "tickfont": {"color": "#94a3b8"},
            "range": [0, 1]
        },
        xaxis={"tickfont": {"color": "#e2e8f0"}},
        font={"color": "#e2e8f0"}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_audit_warnings(audit_flags: list):
    """Render audit warnings panel."""
    if not audit_flags:
        st.success("✅ No quality issues detected!")
        return
    
    st.markdown(f"**{len(audit_flags)} issue(s) found:**")
    for flag in audit_flags:
        if "WARN" in flag:
            st.warning(flag)
        elif "INFO" in flag:
            st.info(flag)
        elif "ERROR" in flag:
            st.error(flag)
        else:
            st.caption(flag)


def render_bloom_badge(bloom_level: str):
    """Render a Bloom's Taxonomy level badge."""
    colors = {
        "Remember": "#6366f1",
        "Understand": "#8b5cf6",
        "Apply": "#06b6d4",
        "Analyze": "#f59e0b",
        "Evaluate": "#f97316",
        "Create": "#ef4444"
    }
    color = colors.get(bloom_level, "#6b7280")
    
    st.markdown(
        f'<div style="display:inline-block; padding:4px 12px; border-radius:12px; '
        f'background-color:{color}; color:white; font-weight:600; font-size:14px;">'
        f'🧠 {bloom_level}</div>',
        unsafe_allow_html=True
    )


def render_semantic_scores(semantic_scores: dict):
    """Render the blended difficulty score breakdown."""
    if not semantic_scores:
        st.info("ℹ️ Semantic scoring unavailable (requires Groq API key).")
        return

    blend = semantic_scores.get("blend", {})
    llm_r = semantic_scores.get("llm_rating", {})
    stu_r = semantic_scores.get("student_sim", {})
    components = blend.get("component_scores", {})

    # Score breakdown bar chart
    labels, values, bar_colors = [], [], []
    if components.get("xgboost") is not None:
        labels.append("XGBoost (surface)")
        values.append(components["xgboost"])
        bar_colors.append("#6366f1")
    if components.get("llm_semantic") is not None:
        labels.append("LLM (semantic)")
        values.append(components["llm_semantic"])
        bar_colors.append("#8b5cf6")
    if components.get("student_proxy") is not None:
        labels.append("Student proxy (1−confidence)")
        values.append(components["student_proxy"])
        bar_colors.append("#06b6d4")

    if labels:
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont={"color": "#e2e8f0"},
        ))
        fig.update_layout(
            title={
                "text": f"Blended Difficulty: {blend.get('blended_difficulty_index', 0):.2f} → {blend.get('blended_difficulty', '?')}",
                "font": {"size": 14, "color": "#e2e8f0"}
            },
            height=180,
            margin=dict(l=10, r=60, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"range": [0, 1], "gridcolor": "rgba(100,116,139,0.2)",
                   "tickfont": {"color": "#94a3b8"}},
            yaxis={"tickfont": {"color": "#e2e8f0"}},
        )
        st.plotly_chart(fig, use_container_width=True)

    # LLM reasoning + student response
    col1, col2 = st.columns(2)
    with col1:
        if llm_r.get("available"):
            st.markdown(f"**🤖 LLM Rating:** {llm_r.get('raw_score', '?')}/10 "
                        f"({llm_r.get('llm_bloom_level', '')})")
            st.caption(f"*\"{llm_r.get('llm_reasoning', '')}\"*")
    with col2:
        if stu_r.get("available"):
            conf_pct = f"{stu_r['student_confidence']:.0%}"
            st.markdown(f"**🧑‍🎓 Student:** selected **{stu_r.get('student_selected', '?')}** "
                        f"with {conf_pct} confidence")
            st.caption(f"*\"{stu_r.get('student_reasoning', '')}\"*")


def render_dashboard(result: dict):
    """Render the full analysis dashboard."""
    ml_stats = result.get("ml_stats", {})
    feature_audit = result.get("feature_audit", [])
    bloom_level = result.get("current_bloom_level", "Unknown")
    semantic_scores = result.get("semantic_scores", {})

    if not ml_stats:
        st.warning("No ML analysis available.")
        return

    # Bloom's level badge
    render_bloom_badge(bloom_level)
    st.divider()

    # Gauges row
    col1, col2 = st.columns(2)
    with col1:
        render_difficulty_gauge(ml_stats)
    with col2:
        render_discrimination_bar(ml_stats)

    # Probabilities
    render_probability_chart(ml_stats)

    # Feature importance
    render_feature_importance(ml_stats)

    # Semantic score breakdown
    with st.expander("🧠 Semantic Score Breakdown (LLM + Student)", expanded=bool(semantic_scores)):
        render_semantic_scores(semantic_scores)

    # Audit warnings
    with st.expander("🔍 Quality Audit", expanded=bool(feature_audit)):
        render_audit_warnings(feature_audit)
