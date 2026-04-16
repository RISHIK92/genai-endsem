"""
RAG Retriever for pedagogical context.

Queries the FAISS vector index to retrieve relevant pedagogical
research and best practices for question improvement.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL, RAG_TOP_K, FAISS_INDEX_DIR


class PedagogyRetriever:
    """Retrieves pedagogically relevant context from the FAISS knowledge base."""

    def __init__(self, index_dir: str = None):
        """Load the FAISS index and metadata.
        
        Args:
            index_dir: Path to directory containing index.faiss and metadata.json.
                       Defaults to FAISS_INDEX_DIR from settings.
        """
        self.index_dir = Path(index_dir) if index_dir else FAISS_INDEX_DIR
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []
        self._loaded = False
        
        self._load()

    def _load(self):
        """Load index and metadata from disk."""
        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.json"
        
        if not index_path.exists():
            print(f"⚠️ FAISS index not found at {index_path}")
            print("   Run the indexer first to build the knowledge base.")
            return
        
        self.index = faiss.read_index(str(index_path))
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        self._loaded = True
        print(f"✅ Knowledge base loaded: {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = None) -> list:
        """Retrieve the most relevant chunks for a query.
        
        Args:
            query: The search query (e.g., "How to improve question with 
                   high LaTeX density but low discrimination?")
            top_k: Number of results to return. Defaults to RAG_TOP_K.
            
        Returns:
            List of dicts: [
                {
                    'text': '...',
                    'source': 'blooms_taxonomy.pdf',
                    'chunk_id': 3,
                    'score': 0.87
                }, ...
            ]
        """
        if not self._loaded or self.index is None or self.index.ntotal == 0:
            return self._fallback_retrieve(query)
        
        top_k = top_k or RAG_TOP_K
        top_k = min(top_k, self.index.ntotal)
        
        # Encode query
        query_embedding = self.embedder.encode(
            [query], 
            normalize_embeddings=True
        ).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", i),
                "score": round(float(dist), 4)
            })
        
        return results

    def _fallback_retrieve(self, query: str) -> list:
        """Provide basic pedagogical context when no index is available."""
        query_lower = query.lower()
        
        fallback_responses = []
        
        if any(w in query_lower for w in ["bloom", "taxonomy", "cognitive", "level"]):
            fallback_responses.append({
                "text": "Bloom's Taxonomy defines six cognitive levels: Remember, Understand, Apply, Analyze, Evaluate, Create. To increase question difficulty, move up the taxonomy by requiring higher-order thinking. Use action verbs specific to each level.",
                "source": "fallback_bloom",
                "chunk_id": 0,
                "score": 0.80
            })
        
        if any(w in query_lower for w in ["distractor", "option", "choice", "mcq"]):
            fallback_responses.append({
                "text": "Good distractors should represent common misconceptions and be plausible to students who haven't mastered the material. Avoid 'All of the above' options. Ensure distractors are similar in length and grammatical structure to the correct answer.",
                "source": "fallback_mcq",
                "chunk_id": 0,
                "score": 0.75
            })
        
        if any(w in query_lower for w in ["discrimination", "item analysis", "irt"]):
            fallback_responses.append({
                "text": "Poor discrimination often results from items that are too easy or too hard, unclear stems, or implausible distractors. To improve discrimination: align with learning objectives, use targeted distractors, and ensure moderate difficulty (P-value 0.3-0.7).",
                "source": "fallback_irt",
                "chunk_id": 0,
                "score": 0.75
            })
        
        if any(w in query_lower for w in ["latex", "formula", "math", "notation"]):
            fallback_responses.append({
                "text": "High LaTeX density can increase difficulty through notation complexity rather than conceptual difficulty. Simplify by: using words instead of symbols where possible, breaking complex expressions into steps, and providing notation guides.",
                "source": "fallback_latex",
                "chunk_id": 0,
                "score": 0.70
            })
        
        if any(w in query_lower for w in ["difficulty", "easy", "hard", "medium"]):
            fallback_responses.append({
                "text": "Question difficulty is influenced by: cognitive level (Bloom's), content complexity, language clarity, option quality, and student familiarity. To make a question easier: reduce cognitive level, simplify language, provide context. To make harder: require multi-step reasoning, add realistic distractors.",
                "source": "fallback_difficulty",
                "chunk_id": 0,
                "score": 0.70
            })
        
        if any(w in query_lower for w in ["critical thinking", "higher order", "analyze"]):
            fallback_responses.append({
                "text": "To test critical thinking with MCQs: present novel scenarios, include data interpretation, ask students to identify assumptions, present multiple valid approaches, use case studies, and require multi-step reasoning.",
                "source": "fallback_critical_thinking",
                "chunk_id": 0,
                "score": 0.70
            })

        # Always add a generic one
        if not fallback_responses:
            fallback_responses.append({
                "text": "Effective assessment design principles: Align questions with learning objectives, ensure validity and reliability, use appropriate cognitive levels from Bloom's Taxonomy, design plausible distractors based on common misconceptions, and maintain clarity in the question stem.",
                "source": "fallback_general",
                "chunk_id": 0,
                "score": 0.50
            })
        
        return fallback_responses

    def format_context(self, results: list) -> str:
        """Format retrieval results into a coherent context string for the LLM.
        
        Args:
            results: List of retrieval result dicts.
            
        Returns:
            Formatted string ready for LLM prompt.
        """
        if not results:
            return "No relevant pedagogical context found."
        
        sections = []
        for i, result in enumerate(results, 1):
            source = result.get("source", "unknown").replace("builtin_", "").replace("_", " ").title()
            sections.append(
                f"[{i}] (Source: {source}, Relevance: {result['score']:.2f})\n"
                f"{result['text']}"
            )
        
        return "\n\n".join(sections)

    def build_targeted_query(self, ml_stats: dict, user_request: str = "") -> str:
        """Build a targeted RAG query from ML statistics.
        
        Constructs a query that will retrieve the most relevant pedagogical
        guidance based on the specific issues identified by the ML model.
        
        Args:
            ml_stats: Output from QuestionPredictor.predict()
            user_request: Optional user request for additional context.
            
        Returns:
            A query string optimized for retrieval.
        """
        parts = []
        
        # Difficulty-based query
        difficulty = ml_stats.get("difficulty", "Medium")
        discrimination = ml_stats.get("discrimination", "Fair")
        parts.append(f"question with {difficulty} difficulty and {discrimination} discrimination")
        
        # Feature-based specifics
        importance = ml_stats.get("feature_importance", {})
        top_features = list(importance.keys())[:3]
        
        feature_descriptions = {
            "latex_density": "high LaTeX density and mathematical notation",
            "word_count": "long word count and verbose stem",
            "stem_length": "lengthy question stem",
            "readability_score": "low readability and complex language",
            "formula_count": "many mathematical formulas",
            "distractor_similarity": "distractor similarity issues",
            "has_negation": "negative phrasing in stem",
            "bloom_level_encoded": "cognitive level mismatch",
            "misconception_count": "misconception patterns",
            "option_length_variance": "unequal option lengths",
        }
        
        for feat in top_features:
            if feat in feature_descriptions:
                parts.append(feature_descriptions[feat])
        
        # User request
        if user_request:
            parts.append(user_request)
        
        query = "How to improve a " + ", ".join(parts)
        return query

    @property
    def is_loaded(self) -> bool:
        """Check if the knowledge base is loaded."""
        return self._loaded
