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
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme enhancement */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.15) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 4px 0 0 0;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 12px !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
        border: 1px solid rgba(99,102,241,0.2);
        color: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.2) !important;
        border-color: #6366f1 !important;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(99,102,241,0.2) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    st.markdown("## ⚙️ Configuration")
    
    # API Key
    api_key = st.text_input(
        "Groq API Key",
        value=GROQ_API_KEY or "",
        type="password",
        help="Get a free API key at https://console.groq.com"
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    st.divider()
    
    # Target settings
    st.markdown("### 🎯 Target Settings")
    
    target_difficulty = st.selectbox(
        "Target Difficulty",
        ["", "Easy", "Medium", "Hard"],
        index=0,
        help="Leave empty for general analysis"
    )
    
    target_bloom = st.selectbox(
        "Target Bloom's Level",
        ["", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
        index=0,
        help="Leave empty for auto-detection"
    )
    
    subject = st.text_input(
        "Subject Area",
        value="General",
        help="e.g., Computer Science, Mathematics, Physics"
    )
    
    st.divider()
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    model_choice = st.radio(
        "XGBoost Model",
        ["Model A (Text Only — 25 features)", "Model B (All Features — 30 features)"],
        index=1
    )
    
    st.divider()
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Rebuilding..."):
            indexer = KnowledgeBaseIndexer()
            indexer.build_and_save(str(PDF_DIR), str(FAISS_INDEX_DIR))
            st.session_state.kb_initialized = True
        st.success("Knowledge base rebuilt!")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analysis_result = None
        st.session_state.debate_result = None
        st.rerun()
    
    st.divider()
    st.caption("Built with ❤️ using LangGraph + XGBoost + FAISS")
    st.caption("Powered by Groq (Llama 3.1 70B)")


# ─── Main Content ────────────────────────────────────────────
# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 EduAgent-OS</h1>
    <p>The Autonomous Assessment Optimizer — Powered by XGBoost, RAG, and LLM Reasoning</p>
</div>
""", unsafe_allow_html=True)

# Initialize KB
initialize_knowledge_base()

# Tabs
tab_chat, tab_analyze, tab_debate = st.tabs([
    "💬 Chat Interface", 
    "📊 Direct Analysis", 
    "🎭 Multi-Agent Debate"
])

# ─── Tab 1: Chat Interface ──────────────────────────────────
with tab_chat:
    # Example queries
    with st.expander("💡 Example queries", expanded=False):
        example_queries = [
            "Analyze this question and suggest improvements",
            "My students failed this question. Explain why and fix it.",
            "Make this question test Critical Thinking instead of Recall",
            "Generate an Apply-level version of this question",
            "This question has poor discrimination. How can I improve it?",
        ]
        for q in example_queries:
            st.caption(f"• *\"{q}\"*")
    
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Paste a question or ask about assessment design..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if API key is set
        if not (api_key or GROQ_API_KEY):
            with st.chat_message("assistant"):
                st.error("⚠️ Please set your Groq API Key in the sidebar to use the chat.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "⚠️ Please set your Groq API Key in the sidebar."
                })
        else:
            # Run the agent pipeline
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing..."):
                    try:
                        result = run_pipeline(
                            question=prompt,
                            user_request=prompt,
                            target_bloom_level=target_bloom,
                            target_difficulty=target_difficulty,
                            subject=subject
                        )
                        
                        st.session_state.analysis_result = result
                        
                        # Display agent messages
                        agent_messages = result.get("messages", [])
                        response_text = "\n\n".join(agent_messages)
                        
                        # Add reasoning (already cleaned of <think> tags by the reasoner node)
                        reasoning = result.get("reasoning", "")
                        if reasoning:
                            response_text += f"\n\n---\n\n### 🧠 Expert Analysis\n\n{reasoning}"
                        
                        st.markdown(response_text)
                        
                        # Show thinking in a collapsible expander (if any)
                        thinking = result.get("thinking", "")
                        if thinking:
                            with st.expander("💭 View Model Thinking Process", expanded=False):
                                st.markdown(thinking)
                        
                        # Show refined questions inline
                        refined = result.get("refined_questions", {})
                        if any(refined.values()):
                            st.divider()
                            render_question_editor(result)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text[:2000]  # Truncate for history
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
    
    # Show dashboard if results exist
    if st.session_state.analysis_result:
        st.divider()
        col_dash, col_report = st.columns([3, 1])
        with col_dash:
            with st.expander("📊 Full Analysis Dashboard", expanded=False):
                render_dashboard(st.session_state.analysis_result)
        with col_report:
            render_report_download(st.session_state.analysis_result)


# ─── Tab 2: Direct Analysis ─────────────────────────────────
with tab_analyze:
    st.subheader("📊 Direct Question Analysis")
    st.caption("Paste a question below for instant ML analysis without LLM reasoning.")
    
    question_input = st.text_area(
        "MCQ Question (include stem and all options)",
        height=200,
        placeholder="""What is the time complexity of binary search?
A) O(n)
B) O(log n)
C) O(n log n)
D) O(n²)"""
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and question_input.strip():
        with st.spinner("Running XGBoost analysis..."):
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
    
    # Full pipeline button
    if question_input.strip() and (api_key or GROQ_API_KEY):
        st.divider()
        if st.button("🚀 Run Full Agent Pipeline (Analyze + Reason + Refine)", 
                      use_container_width=True):
            with st.spinner("🤖 Running full pipeline... This may take 30-60 seconds."):
                try:
                    result = run_pipeline(
                        question=question_input,
                        user_request="Analyze and improve this question",
                        target_bloom_level=target_bloom,
                        target_difficulty=target_difficulty,
                        subject=subject
                    )
                    
                    st.session_state.analysis_result = result
                    
                    # Display results
                    render_dashboard(result)
                    
                    reasoning = result.get("reasoning", "")
                    if reasoning:
                        st.divider()
                        st.subheader("🧠 Expert Reasoning")
                        st.markdown(reasoning)
                    
                    # Show thinking in a collapsible expander (if any)
                    thinking = result.get("thinking", "")
                    if thinking:
                        with st.expander("💭 View Model Thinking Process", expanded=False):
                            st.markdown(thinking)
                    
                    refined = result.get("refined_questions", {})
                    if any(refined.values()):
                        st.divider()
                        render_question_editor(result)
                        render_original_vs_refined(
                            question_input, refined, 
                            result.get("validation_results", {})
                        )
                    
                    # Report download
                    render_report_download(result)
                    
                except Exception as e:
                    st.error(f"Pipeline error: {str(e)}")


# ─── Tab 3: Multi-Agent Debate ──────────────────────────────
with tab_debate:
    st.subheader("🎭 Multi-Agent Synthetic Student Debate")
    st.caption(
        "Three AI agents collaborate to design the optimal question: "
        "a Professor, an ML Predictor, and a Student Persona."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        debate_topic = st.text_input(
            "Topic / Concept",
            placeholder="e.g., Binary Search, Newton's Laws"
        )
    
    with col2:
        debate_bloom = st.selectbox(
            "Bloom's Level",
            ["Apply", "Analyze", "Evaluate", "Remember", "Understand", "Create"],
            key="debate_bloom"
        )
    
    with col3:
        debate_difficulty = st.selectbox(
            "Target Difficulty",
            ["Medium", "Easy", "Hard"],
            key="debate_difficulty"
        )
    
    max_rounds = st.slider("Max Debate Rounds", 1, 5, 3)
    
    if st.button("🎭 Start Debate", type="primary", use_container_width=True):
        if not debate_topic:
            st.warning("Please enter a topic.")
        elif not (api_key or GROQ_API_KEY):
            st.error("Please set your Groq API Key in the sidebar.")
        else:
            with st.spinner("🎭 Debate in progress... Agents are discussing..."):
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
        
        # Summary
        consensus_icon = "✅" if result["consensus_reached"] else "⚠️"
        col1, col2, col3 = st.columns(3)
        col1.metric("Rounds", result["rounds_taken"])
        col2.metric("Consensus", f"{consensus_icon} {'Yes' if result['consensus_reached'] else 'No'}")
        
        report = result.get("difficulty_report", {})
        predicted = report.get("prediction", {}).get("difficulty", "?")
        col3.metric("Final Difficulty", predicted)
        
        st.divider()
        
        # Final question
        st.subheader("🏆 Final Question")
        st.markdown(result["final_question"])
        
        # Debate transcript
        st.divider()
        st.subheader("📜 Debate Transcript")
        
        for round_entry in result["debate_transcript"]:
            with st.expander(f"Round {round_entry['round']}", expanded=round_entry['round'] == 1):
                for exchange in round_entry["exchanges"]:
                    agent_icons = {
                        "Professor": "👨‍🏫",
                        "ML Predictor": "🤖",
                    }
                    icon = agent_icons.get(exchange["agent"], "🧑‍🎓")
                    
                    st.markdown(f"**{icon} {exchange['agent']}** — *{exchange['action']}*")
                    st.markdown(exchange["output"])
                    
                    # Show thinking in a nested expander if present
                    thinking = exchange.get("thinking", "")
                    if thinking:
                        with st.expander(f"💭 {exchange['agent']} Thinking", expanded=False):
                            st.markdown(thinking)
                    
                    st.divider()
