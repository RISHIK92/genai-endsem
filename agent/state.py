"""
Agent State Definition for LangGraph.

Defines the shared state schema that flows through all nodes
in the assessment optimizer pipeline.
"""
from typing import TypedDict, Annotated, Any
from operator import add


class AgentState(TypedDict):
    """State schema for the EduAgent-OS LangGraph state machine.
    
    This state is shared across all nodes and updated incrementally
    as the question flows through the pipeline.
    """
    
    # ─── Input ───────────────────────────────────────────────
    raw_question: str              # Original MCQ text (stem + options)
    user_request: str              # Natural language request from user
    target_bloom_level: str        # Target Bloom's level (e.g., "Apply")
    target_difficulty: str         # Target difficulty (Easy/Medium/Hard)
    subject: str                   # Subject area (e.g., "Computer Science")
    
    # ─── Node 1: Analyst Output ──────────────────────────────
    ml_stats: dict                 # XGBoost predictions + SHAP (blended with semantic)
    feature_audit: list            # Warnings from feature extractor
    current_bloom_level: str       # Detected Bloom's level of original
    semantic_scores: dict          # LLM difficulty rating + student sim scores
    
    # ─── Node 2: Pedagogy Retriever Output ───────────────────
    pedagogy_context: str          # Retrieved RAG chunks (formatted)
    retrieval_sources: list        # Source attribution list
    rag_query: str                 # The query used for retrieval
    
    # ─── Node 3: Reasoner Output ────────────────────────────
    reasoning: str                 # LLM's analysis and rationale (clean, no <think> tags)
    improvement_strategy: str      # Concrete action plan
    thinking: str                  # LLM's internal thinking process (shown in expander)
    
    # ─── Node 4: Refiner Output ─────────────────────────────
    refined_questions: dict        # {"easy": ..., "medium": ..., "hard": ...}
    difficulty_justification: str  # Why each version has its difficulty
    validation_results: dict       # XGBoost re-validation of generated Qs
    
    # ─── Control Flow ───────────────────────────────────────
    critique: str                  # Self-critique for iteration
    iteration_count: int           # Track refinement loops
    should_continue: bool          # Whether to loop back to reasoner
    error: str                     # Error message if any node fails
    
    # ─── Conversation History ───────────────────────────────
    messages: Annotated[list, add] # Accumulated messages for chat display
