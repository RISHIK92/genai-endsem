"""
EduAgent-OS — The Autonomous Assessment Optimizer

Main Streamlit application entry point.
Features:
- Chat interface for natural language interaction
- ML-powered question analysis dashboard
- Side-by-side refined question comparison
- Multi-agent debate system
- Downloadable reports
"""
import streamlit as st
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import run_pipeline
from debate.debate_runner import DebateRunner
from rag.indexer import KnowledgeBaseIndexer
from ui.analysis_dashboard import render_dashboard
from ui.question_editor import render_question_editor, render_original_vs_refined
from ui.report_generator import render_report_download
from config.settings import FAISS_INDEX_DIR, PDF_DIR, GROQ_API_KEY
from utils.think_parser import separate_thinking


# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduAgent-OS | Assessment Optimizer",
    page_icon="🎓",
    layout="wide",
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Root & typography ─────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #e4e4e7 !important;
    }

    /* ── App background ─────────────────────── */
    .stApp {
        background-color: #09090b;
    }

    /* ── Hide Streamlit chrome ──────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 1200px; }

    /* ── Professional Header ────────────────── */
    .hero-header {
        background: #141414;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 32px 40px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .hero-title {
        color: #fafafa;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        color: #a1a1aa;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0 0 16px 0;
    }
    .hero-description {
        color: #71717a;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
        max-width: 800px;
    }

    /* ── Sidebar ────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #0f0f11 !important;
        border-right: 1px solid #27272a !important;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 16px 0 20px 0;
        border-bottom: 1px solid #27272a;
        margin-bottom: 20px;
    }
    .sidebar-logo-text {
        font-size: 1rem;
        font-weight: 700;
        color: #fafafa;
        letter-spacing: 0.02em;
    }
    .sidebar-section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #71717a;
        margin: 20px 0 8px 2px;
    }

    /* ── Inputs ─────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stPassword > div > div > input {
        background: #141414 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        color: #e4e4e7 !important;
        padding: 10px 12px !important;
        box-shadow: none !important;
    }

    /* ── Selectbox: force ALL text inside to be visible ── */
    [data-baseweb="select"] *,
    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="select"] p,
    [data-baseweb="select"] [class*="singleValue"],
    [data-baseweb="select"] [class*="placeholder"],
    [data-baseweb="select"] [class*="ValueContainer"] * {
        color: #e4e4e7 !important;
        -webkit-text-fill-color: #e4e4e7 !important;
        background-color: transparent !important;
    }

    /* Container background */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] [class*="control"],
    [data-baseweb="select"] [class*="container"] {
        background-color: #141414 !important;
        border-color: #27272a !important;
    }

    /* Dropdown menu */
    [data-baseweb="menu"],
    [data-baseweb="menu"] * {
        background-color: #141414 !important;
        color: #e4e4e7 !important;
        -webkit-text-fill-color: #e4e4e7 !important;
    }
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [aria-selected="true"] {
        background-color: #27272a !important;
        color: #fafafa !important;
        -webkit-text-fill-color: #fafafa !important;
    }
    [data-baseweb="select"] svg {
        fill: #a1a1aa !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder,
    .stPassword > div > div > input::placeholder {
        color: #52525b !important;
        opacity: 1 !important;
    }
    .stChatInput textarea::placeholder {
        color: #52525b !important;
        opacity: 1 !important;
    }
    .stChatInput { border-color: #27272a !important; }
    label, .stRadio label span, .stSlider label {
        color: #a1a1aa !important;
    }

    /* ── Buttons ─────────────────────────────── */
    .stButton > button {
        background: #fafafa !important;
        color: #09090b !important;
        border: 1px solid #fafafa !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #e4e4e7 !important;
        border-color: #e4e4e7 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Tabs ────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0;
        border-bottom: 1px solid #27272a;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 0 !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: #71717a !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #fafafa !important;
        border-bottom: 2px solid #fafafa !important;
    }

    /* ── Chat & Markdown ────────────────────── */
    .stChatMessage {
        background: #141414 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }
    .stMarkdown p, .stMarkdown li {
        line-height: 1.8 !important;
        color: #d4d4d8 !important;
        font-size: 0.95rem !important;
    }
    .stMarkdown ul, .stMarkdown ol {
        margin-bottom: 1.2em !important;
        padding-left: 1.5em !important;
    }
    .stMarkdown strong, .stMarkdown b {
        color: #fafafa !important;
        font-weight: 700 !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #fafafa !important;
        margin-top: 1.5em !important;
        margin-bottom: 0.5em !important;
        font-weight: 600 !important;
    }
    .stMarkdown code {
        background: rgba(63,63,70,0.5) !important;
        border: 1px solid #3f3f46 !important;
        color: #a78bfa !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.85em !important;
    }
    .stCodeBlock {
        background: #141414 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        margin: 16px 0 !important;
    }
    .stMarkdown blockquote {
        border-left: 4px solid #3b82f6 !important;
        color: #a1a1aa !important;
        background: rgba(59,130,246,0.05) !important;
        padding: 12px 16px !important;
        margin: 16px 0 !important;
        border-radius: 0 8px 8px 0 !important;
    }

    /* ── Cards & Sections ────────────────────── */
    .glass-card {
        background: #141414;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        margin-bottom: 16px;
    }
    .section-heading {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fafafa;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-sub {
        color: #71717a;
        font-size: 0.9rem;
        margin: 0 0 20px 0;
        line-height: 1.5;
    }

    /* ── Metrics ─────────────────────────────── */
    [data-testid="stMetric"] {
        background: #141414 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        padding: 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stMetricValue"] {
        color: #fafafa !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #71717a !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* ── Expanders ───────────────────────────── */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background: #141414 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        color: #d4d4d8 !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }

    /* ── Alerts ──────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 6px !important;
        border: 1px solid #27272a !important;
        background: #141414 !important;
    }

    /* ── Agent Badges ────────────────────────── */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 4px;
        background: #1c1c1e;
        border: 1px solid #27272a;
        color: #a1a1aa;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .agent-badge-dot { font-size: 0.9rem; }

    hr { border-color: #27272a !important; margin: 24px 0 !important; }

    /* ── Misc ─────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #71717a !important;
        font-size: 0.78rem !important;
    }
    h2, h3 {
        color: #fafafa !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Download button ──────────────────────── */
    .stDownloadButton > button {
        background: #fafafa !important;
        color: #09090b !important;
        border: 1px solid #fafafa !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    .stDownloadButton > button:hover {
        background: #e4e4e7 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }

    /* ── Spinner ─────────────────────────────── */
    .stSpinner > div > div {
        border-top-color: #3b82f6 !important;
    }

    /* ── Query pills ─────────────────────────── */
    .query-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }
    .query-pill {
        display: inline-block;
        padding: 6px 14px;
        background: #1c1c1e;
        border: 1px solid #27272a;
        border-radius: 99px;
        font-size: 0.82rem;
        color: #a1a1aa;
        font-weight: 500;
    }

    /* ── Divider ─────────────────────────────── */
    [data-testid="stHorizontalRule"] {
        border-color: #27272a !important;
    }

    /* ── Scrollbar ───────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f0f11; }
    ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #52525b; }

    /* ── Force sidebar always open ───────────── */
    section[data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        display: block !important;
        min-width: 280px !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "debate_result" not in st.session_state:
    st.session_state.debate_result = None

if "kb_initialized" not in st.session_state:
    st.session_state.kb_initialized = False


def initialize_knowledge_base():
    """Build FAISS index if not already done."""
    if not st.session_state.kb_initialized:
        index_path = FAISS_INDEX_DIR / "index.faiss"
        if not index_path.exists():
            with st.spinner("🔨 Building knowledge base (first run only)..."):
                indexer = KnowledgeBaseIndexer()
                indexer.build_and_save(str(PDF_DIR), str(FAISS_INDEX_DIR))
        st.session_state.kb_initialized = True


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-text">EduAgent-OS</span>
    </div>
    """, unsafe_allow_html=True)

    # Read key at runtime so st.secrets is available (Streamlit Cloud)
    import os as _os
    try:
        import streamlit as _st
        api_key = _st.secrets.get("GROQ_API_KEY", _os.getenv("GROQ_API_KEY", ""))
    except Exception:
        api_key = _os.getenv("GROQ_API_KEY", "")

    st.divider()

    # Target settings
    st.markdown('<div class="sidebar-section-label">Target Settings</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.82rem;color:#a1a1aa;margin-bottom:4px;">Difficulty</div>', unsafe_allow_html=True)
    target_difficulty_raw = st.radio(
        "Difficulty",
        ["Any", "Easy", "Medium", "Hard"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help="Leave on Any for general analysis"
    )
    target_difficulty = "" if target_difficulty_raw == "Any" else target_difficulty_raw

    st.markdown('<div style="font-size:0.82rem;color:#a1a1aa;margin-bottom:4px;margin-top:8px;">Bloom\'s Level</div>', unsafe_allow_html=True)
    target_bloom_raw = st.radio(
        "Bloom's Level",
        ["Any", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help="Leave on Any for auto-detection"
    )
    target_bloom = "" if target_bloom_raw == "Any" else target_bloom_raw

    subject = st.text_input(
        "Subject Area",
        value="General",
        placeholder="e.g., Computer Science",
        help="e.g., Computer Science, Mathematics, Physics"
    )

    st.divider()

    # Model selection
    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "XGBoost Model",
        ["Model A — Text features (25)", "Model B — All features (30)"],
        index=1,
        label_visibility="collapsed"
    )

    st.divider()

    # Quick actions
    st.markdown('<div class="sidebar-section-label">Actions</div>', unsafe_allow_html=True)

    if st.button("Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            indexer = KnowledgeBaseIndexer()
            indexer.build_and_save(str(PDF_DIR), str(FAISS_INDEX_DIR))
            st.session_state.kb_initialized = True
        st.success("Knowledge base rebuilt.")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analysis_result = None
        st.session_state.debate_result = None
        st.rerun()

    st.divider()

    # System info
    st.markdown("""
    <div style="padding: 16px; background: #141414; border: 1px solid #27272a; border-radius: 8px;">
        <div style="font-size: 0.7rem; color: #52525b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;">System Stack</div>
        <div style="font-size: 0.82rem; color: #71717a; line-height: 2;">
            <span style="color:#a1a1aa;font-weight:600;">LangGraph</span> &nbsp;·&nbsp; multi-agent pipeline<br>
            <span style="color:#a1a1aa;font-weight:600;">XGBoost</span> &nbsp;·&nbsp; difficulty classifier<br>
            <span style="color:#a1a1aa;font-weight:600;">FAISS</span> &nbsp;·&nbsp; pedagogical RAG<br>
            <span style="color:#a1a1aa;font-weight:600;">Groq</span> &nbsp;·&nbsp; Llama 3.1 70B<br>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ────────────────────────────────────────────
# Hero Header
st.markdown("""
<div class="hero-header">
    <p class="hero-title">EduAgent-OS</p>
    <p class="hero-subtitle">Autonomous Assessment Optimizer</p>
    <p class="hero-description">Enterprise-grade question analysis powered by XGBoost, FAISS-backed RAG, and LLM multi-agent reasoning. Analyze, audit, and systematically improve MCQs for maximum pedagogical effectiveness.</p>
</div>
""", unsafe_allow_html=True)

# Initialize KB
initialize_knowledge_base()

# ── Feature Info Box ──────────────────────────────────────────
st.markdown("""
<div style="background:#141414;border:1px solid #27272a;border-radius:8px;padding:20px 24px;margin-bottom:20px;">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#52525b;margin-bottom:14px;">Feature Overview</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
        <div style="background:#0f0f11;border:1px solid #27272a;border-radius:6px;padding:14px;">
            <div style="font-size:0.8rem;font-weight:700;color:#fafafa;margin-bottom:6px;letter-spacing:0.02em;">Chat</div>
            <div style="font-size:0.8rem;color:#71717a;line-height:1.65;">Paste any MCQ and converse naturally. Ask the agent to <span style="color:#a1a1aa;">analyze</span>, <span style="color:#a1a1aa;">rewrite</span>, elevate Bloom's level, fix distractors, or diagnose poor student performance.</div>
        </div>
        <div style="background:#0f0f11;border:1px solid #27272a;border-radius:6px;padding:14px;">
            <div style="font-size:0.8rem;font-weight:700;color:#fafafa;margin-bottom:6px;letter-spacing:0.02em;">Direct Analysis</div>
            <div style="font-size:0.8rem;color:#71717a;line-height:1.65;"><span style="color:#a1a1aa;">Analyze</span> — instant XGBoost difficulty + Bloom's scoring, no LLM required.<br><span style="color:#a1a1aa;">Full Pipeline</span> — runs all 4 agent nodes: Analyst → RAG → Reasoner → Refiner.</div>
        </div>
        <div style="background:#0f0f11;border:1px solid #27272a;border-radius:6px;padding:14px;">
            <div style="font-size:0.8rem;font-weight:700;color:#fafafa;margin-bottom:6px;letter-spacing:0.02em;">Agent Debate</div>
            <div style="font-size:0.8rem;color:#71717a;line-height:1.65;"><span style="color:#a1a1aa;">Create</span> a new question from scratch. Three AI personas — Professor, ML Predictor, and Student Simulator — debate iteratively until reaching consensus.</div>
        </div>
    </div>
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid #27272a;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
        <div style="font-size:0.78rem;color:#a1a1aa;"><span style="color:#fafafa;font-weight:600;display:block;margin-bottom:2px;">Difficulty</span>Target Easy / Medium / Hard. XGBoost validates the output matches the goal.</div>
        <div style="font-size:0.78rem;color:#a1a1aa;"><span style="color:#fafafa;font-weight:600;display:block;margin-bottom:2px;">Bloom's Level</span>Shift cognitive demand from Remember up through Create.</div>
        <div style="font-size:0.78rem;color:#a1a1aa;"><span style="color:#fafafa;font-weight:600;display:block;margin-bottom:2px;">Subject Area</span>Scopes RAG retrieval to domain-specific pedagogy guidelines.</div>
        <div style="font-size:0.78rem;color:#a1a1aa;"><span style="color:#fafafa;font-weight:600;display:block;margin-bottom:2px;">XGBoost Model</span>Model A: 25 text features. Model B: 30 features including structural signals.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab_chat, tab_analyze, tab_debate = st.tabs([
    "Chat",
    "Direct Analysis",
    "Agent Debate",
])


# ─── Tab 1: Chat Interface ────────────────────────────────────
with tab_chat:
    # Example queries card
    example_queries = [
        "Analyze and improve this question",
        "Make this test critical thinking, not recall",
        "My students failed this — explain why and fix it",
        "Generate an Apply-level version of this question",
        "This question has poor discrimination. How do I improve it?",
    ]

    st.markdown("""
    <div class="glass-card" style="margin-bottom: 12px;">
        <div class="section-heading">Try asking…</div>
        <div class="query-pills">
    """ + "".join([f'<span class="query-pill">{q}</span>' for q in example_queries]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Paste a question or ask about assessment design…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not api_key:
            with st.chat_message("assistant"):
                st.error("No API key found. Set GROQ_API_KEY in your environment.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No API key configured."
                })
        else:
            with st.chat_message("assistant"):
                with st.spinner("Running agent pipeline…"):
                    try:
                        result = run_pipeline(
                            question=prompt,
                            user_request=prompt,
                            target_bloom_level=target_bloom,
                            target_difficulty=target_difficulty,
                            subject=subject
                        )

                        st.session_state.analysis_result = result

                        agent_messages = result.get("messages", [])
                        raw_response_text = "\n\n".join(agent_messages)
                        
                        clean_text, parsed_think = separate_thinking(raw_response_text)
                        thinking = result.get("thinking", "") or parsed_think
                        reasoning = result.get("reasoning", "")
                        
                        if reasoning:
                            st.info(f"**Expert Analysis & Reasoning**\n\n{reasoning}")

                        if thinking:
                            with st.expander("View Model Thinking Process", expanded=False):
                                st.markdown(thinking)
                                
                        if clean_text.strip():
                            st.markdown(f"**Final Output:**\n\n{clean_text}")

                        refined = result.get("refined_questions", {})
                        if any(refined.values()):
                            st.divider()
                            render_question_editor(result)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": (f"**🧠 Expert Analysis:**\n{reasoning}\n\n" if reasoning else "") + clean_text[:2000]
                        })

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    # Dashboard at bottom if results exist
    if st.session_state.analysis_result:
        st.divider()
        col_dash, col_report = st.columns([3, 1])
        with col_dash:
            with st.expander("Full Analysis Dashboard", expanded=False):
                render_dashboard(st.session_state.analysis_result)
        with col_report:
            render_report_download(st.session_state.analysis_result)


# ─── Tab 2: Direct Analysis ───────────────────────────────────
with tab_analyze:
    st.markdown("""
    <div class="section-heading">Direct Question Analysis</div>
    <p class="section-sub">Paste any MCQ below for instant XGBoost-powered difficulty and Bloom's analysis.</p>
    """, unsafe_allow_html=True)

    question_input = st.text_area(
        "MCQ Question",
        height=180,
        placeholder="""What is the time complexity of binary search?
A) O(n)
B) O(log n)
C) O(n log n)
D) O(n²)""",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
    with col2:
        full_pipeline_btn = st.button(
            "Full Pipeline",
            use_container_width=True,
            disabled=not bool(question_input.strip() and api_key)
        )

    if analyze_btn and question_input.strip():
        with st.spinner("Running XGBoost analysis…"):
            try:
                from ml.predictor import QuestionPredictor
                from ml.feature_extractor import FeatureExtractor

                predictor = QuestionPredictor()
                extractor = FeatureExtractor()

                prediction = predictor.predict(question_input, model="B")
                audit = extractor.audit(question_input)
                bloom = extractor.get_bloom_level(question_input)

                result = {
                    "ml_stats": prediction,
                    "feature_audit": audit,
                    "current_bloom_level": bloom,
                    "raw_question": question_input,
                }

                st.session_state.analysis_result = result
                render_dashboard(result)

            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

    if full_pipeline_btn and question_input.strip():
        with st.spinner("🤖 Running full agent pipeline… This may take 30–60 s."):
            try:
                result = run_pipeline(
                    question=question_input,
                    user_request="Analyze and improve this question",
                    target_bloom_level=target_bloom,
                    target_difficulty=target_difficulty,
                    subject=subject
                )

                st.session_state.analysis_result = result
                render_dashboard(result)

                if reasoning:
                    st.divider()
                    st.markdown('<div class="section-heading">Expert Reasoning</div>', unsafe_allow_html=True)
                    st.markdown(reasoning)

                thinking = result.get("thinking", "")
                if thinking:
                    with st.expander("View Model Thinking Process", expanded=False):
                        st.markdown(thinking)

                refined = result.get("refined_questions", {})
                if any(refined.values()):
                    st.divider()
                    render_question_editor(result)
                    render_original_vs_refined(
                        question_input, refined,
                        result.get("validation_results", {})
                    )

                render_report_download(result)

            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")


# ─── Tab 3: Multi-Agent Debate ────────────────────────────────
with tab_debate:
    st.markdown("""
    <div class="section-heading">Multi-Agent Debate</div>
    <p class="section-sub">Three AI personas — Professor, ML Predictor, and Student Simulator — collaborate to craft the optimal question.</p>
    """, unsafe_allow_html=True)

    # Debate config glass card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        debate_topic = st.text_input(
            "Topic / Concept",
            placeholder="e.g., Binary Search, Newton's Laws"
        )

    with col2:
        st.markdown('<div style="font-size:0.82rem;color:#a1a1aa;margin-bottom:4px;">Bloom\'s Level</div>', unsafe_allow_html=True)
        debate_bloom = st.radio(
            "Bloom's Level",
            ["Apply", "Analyze", "Evaluate", "Remember", "Understand", "Create"],
            index=0,
            key="debate_bloom",
            label_visibility="collapsed"
        )

    with col3:
        st.markdown('<div style="font-size:0.82rem;color:#a1a1aa;margin-bottom:4px;">Target Difficulty</div>', unsafe_allow_html=True)
        debate_difficulty = st.radio(
            "Target Difficulty",
            ["Medium", "Easy", "Hard"],
            index=0,
            key="debate_difficulty",
            label_visibility="collapsed"
        )

    max_rounds = st.slider("Max Debate Rounds", 1, 5, 3,
                           help="More rounds = more refined question, but takes longer")
    st.markdown('</div>', unsafe_allow_html=True)

    debate_btn = st.button("Start Debate", type="primary", use_container_width=True)

    if debate_btn:
        if not debate_topic:
            st.warning("Please enter a topic to start the debate.")
        elif not api_key:
            st.error("No API key configured. Set GROQ_API_KEY in your environment.")
        else:
            with st.spinner("Agents are deliberating. Please wait…"):
                try:
                    runner = DebateRunner()
                    debate_result = runner.run_debate(
                        topic=debate_topic,
                        bloom_level=debate_bloom,
                        target_difficulty=debate_difficulty,
                        max_rounds=max_rounds,
                        subject=subject
                    )
                    st.session_state.debate_result = debate_result
                except Exception as e:
                    st.error(f"Debate error: {str(e)}")

    # Display debate results
    if st.session_state.debate_result:
        result = st.session_state.debate_result

        st.divider()

        # Summary metrics
        consensus_icon = "✅" if result["consensus_reached"] else "⚠️"
        report = result.get("difficulty_report", {})
        predicted = report.get("prediction", {}).get("difficulty", "?")

        m1, m2, m3 = st.columns(3)
        m1.metric("Debate Rounds", result["rounds_taken"])
        m2.metric("Consensus", f"{consensus_icon} {'Yes' if result['consensus_reached'] else 'No'}")
        m3.metric("Final Difficulty", predicted)

        st.divider()

        # Final question highlight
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(16,185,129,0.25);">
            <div class="section-heading" style="color: #34d399;">Final Question</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(result["final_question"])

        st.divider()

        # Debate transcript
        st.markdown('<div class="section-heading">Debate Transcript</div>', unsafe_allow_html=True)

        for round_entry in result["debate_transcript"]:
            with st.expander(
                f"Round {round_entry['round']}",
                expanded=(round_entry["round"] == 1)
            ):
                for exchange in round_entry["exchanges"]:
                    agent_icons = {
                        "Professor": "Prof",
                        "ML Predictor": "ML",
                    }
                    icon = agent_icons.get(exchange["agent"], "Sim")

                    st.markdown(f"""
<div class="agent-badge">
    <span class="agent-badge-dot">{icon}</span>
    {exchange['agent']} &nbsp;·&nbsp; <em style="font-weight:400;color:#71717a;">{exchange['action']}</em>
</div>
""", unsafe_allow_html=True)
                    st.markdown(exchange["output"])

                    thinking = exchange.get("thinking", "")
                    if thinking:
                        with st.expander(f"{exchange['agent']} — Thinking Process", expanded=False):
                            st.markdown(thinking)

                    st.divider()
