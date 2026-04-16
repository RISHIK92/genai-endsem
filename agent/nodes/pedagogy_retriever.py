"""
Node 2: The Pedagogy Retriever

Uses FAISS-backed RAG to retrieve relevant pedagogical research
and best practices based on the ML analysis results.
"""
from rag.retriever import PedagogyRetriever

# Shared instance
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = PedagogyRetriever()
    return _retriever


def pedagogy_retriever_node(state: dict) -> dict:
    """Retrieve relevant pedagogical context from the knowledge base.
    
    Inputs from state:
        - ml_stats: XGBoost prediction results
        - user_request: User's natural language request
        - feature_audit: Audit warnings
        
    Outputs to state:
        - pedagogy_context: Formatted context string for LLM
        - retrieval_sources: Source attribution list
        - rag_query: The query used for retrieval
        - messages: Status message
    """
    ml_stats = state.get("ml_stats", {})
    user_request = state.get("user_request", "")
    feature_audit = state.get("feature_audit", [])
    
    retriever = _get_retriever()
    
    try:
        # Build a targeted query from ML stats
        query = retriever.build_targeted_query(ml_stats, user_request)
        
        # Enrich query with audit findings
        audit_topics = []
        for flag in feature_audit:
            if "LaTeX" in flag:
                audit_topics.append("LaTeX complexity")
            if "readability" in flag.lower():
                audit_topics.append("readability improvement")
            if "Bloom" in flag:
                audit_topics.append("Bloom's taxonomy upgrade")
            if "distractor" in flag.lower():
                audit_topics.append("distractor quality")
            if "negation" in flag.lower():
                audit_topics.append("negative phrasing")
        
        if audit_topics:
            query += " focusing on " + ", ".join(set(audit_topics))
        
        # Retrieve relevant chunks
        results = retriever.retrieve(query, top_k=5)
        
        # Format for LLM consumption
        context = retriever.format_context(results)
        
        # Extract source list for attribution
        sources = [
            {"source": r["source"], "score": r["score"]}
            for r in results
        ]
        
        # Status message
        status = (
            f"📚 **Pedagogy Research Retrieved**\n"
            f"- Query: *\"{query[:100]}...\"*\n"
            f"- Sources found: {len(results)}\n"
            f"- Top sources: {', '.join(s['source'] for s in sources[:3])}"
        )
        
        return {
            "pedagogy_context": context,
            "retrieval_sources": sources,
            "rag_query": query,
            "messages": [status],
        }
        
    except Exception as e:
        error_msg = f"RAG retrieval error: {str(e)}"
        return {
            "pedagogy_context": "Unable to retrieve pedagogical context. Using general assessment design principles.",
            "retrieval_sources": [],
            "rag_query": "",
            "messages": [f"⚠️ {error_msg} — proceeding with general knowledge"],
        }
