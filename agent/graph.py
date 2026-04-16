"""
LangGraph State Machine Assembly.

Connects all four nodes (Analyst → Pedagogy Retriever → Reasoner → Refiner)
with conditional routing for iterative refinement.
"""
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes.analyst import analyst_node
from agent.nodes.pedagogy_retriever import pedagogy_retriever_node
from agent.nodes.reasoner import reasoner_node
from agent.nodes.refiner import refiner_node
from config.settings import MAX_REFINEMENT_ITERATIONS


def should_continue(state: AgentState) -> str:
    """Routing function: decide whether to loop back or finish.
    
    Returns:
        "reasoner" to loop back for another iteration.
        "end" to finish the pipeline.
    """
    if (state.get("should_continue", False) and 
        state.get("iteration_count", 0) < MAX_REFINEMENT_ITERATIONS):
        return "reasoner"
    return "end"


def build_graph():
    """Build and compile the LangGraph state machine.
    
    Pipeline:
        START → Analyst → Pedagogy Retriever → Reasoner → Refiner → [conditional]
                                                    ↑                    ↓
                                                    └──── (if mismatch) ─┘
                                                         (else) → END
    
    Returns:
        Compiled LangGraph runnable.
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("analyst", analyst_node)
    graph.add_node("pedagogy_retriever", pedagogy_retriever_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("refiner", refiner_node)
    
    # Define edges (linear flow)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "pedagogy_retriever")
    graph.add_edge("pedagogy_retriever", "reasoner")
    graph.add_edge("reasoner", "refiner")
    
    # Conditional edge: refiner can loop back to reasoner or finish
    graph.add_conditional_edges(
        "refiner",
        should_continue,
        {
            "reasoner": "reasoner",
            "end": END
        }
    )
    
    # Compile the graph
    compiled = graph.compile()
    return compiled


def run_pipeline(
    question: str,
    user_request: str = "",
    target_bloom_level: str = "",
    target_difficulty: str = "",
    subject: str = "General"
) -> dict:
    """Run the full assessment optimization pipeline.
    
    This is the main entry point for the agent system.
    
    Args:
        question: Raw MCQ text (stem + options).
        user_request: Natural language request from user.
        target_bloom_level: Target Bloom's level (e.g., "Apply").
        target_difficulty: Target difficulty (Easy/Medium/Hard).
        subject: Subject area.
        
    Returns:
        Final AgentState with all results.
    """
    graph = build_graph()
    
    initial_state = {
        "raw_question": question,
        "user_request": user_request,
        "target_bloom_level": target_bloom_level,
        "target_difficulty": target_difficulty,
        "subject": subject,
        "ml_stats": {},
        "feature_audit": [],
        "current_bloom_level": "",
        "semantic_scores": {},
        "pedagogy_context": "",
        "retrieval_sources": [],
        "rag_query": "",
        "reasoning": "",
        "improvement_strategy": "",
        "refined_questions": {},
        "difficulty_justification": "",
        "validation_results": {},
        "critique": "",
        "iteration_count": 0,
        "should_continue": False,
        "error": "",
        "messages": [],
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    return result


# Convenience alias
run = run_pipeline
