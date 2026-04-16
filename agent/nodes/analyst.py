"""
Node 1: The Analyst

Takes the raw question and passes it through the XGBoost models
to produce difficulty/discrimination predictions and feature audit.
"""
from ml.feature_extractor import FeatureExtractor
from ml.predictor import QuestionPredictor
from ml.semantic_scorer import (
    rate_difficulty_with_llm,
    simulate_student_confidence,
    blend_difficulty_scores,
)
from config.settings import GROQ_API_KEY
import os

# Shared instances (loaded once)
_extractor = None
_predictor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = QuestionPredictor()
    return _predictor


def analyst_node(state: dict) -> dict:
    """Analyze the raw question using XGBoost models.
    
    Inputs from state:
        - raw_question: The MCQ text to analyze
        
    Outputs to state:
        - ml_stats: Full prediction results
        - feature_audit: List of audit warnings
        - current_bloom_level: Detected Bloom's level
        - messages: Status message
    """
    raw_question = state.get("raw_question", "")
    
    if not raw_question.strip():
        return {
            "ml_stats": {},
            "feature_audit": ["❌ ERROR: No question provided"],
            "current_bloom_level": "Unknown",
            "messages": ["❌ No question text provided for analysis."],
            "error": "No question text provided"
        }
    
    extractor = _get_extractor()
    predictor = _get_predictor()
    
    try:
        # Run prediction with Model A (text features)
        prediction_a = predictor.predict(raw_question, model="A")
        
        # Run prediction with Model B (all features) 
        prediction_b = predictor.predict(raw_question, model="B")
        
        # Combine results — use Model B as primary, Model A as secondary
        ml_stats = {
            "model_a": prediction_a,
            "model_b": prediction_b,
            # Use Model B's classification as the primary prediction
            "difficulty": prediction_b.get("difficulty", prediction_a.get("difficulty", "Unknown")),
            "difficulty_index": prediction_b.get("difficulty_index", prediction_a.get("difficulty_index", 0.5)),
            "difficulty_probabilities": prediction_b.get("difficulty_probabilities", {}),
            "discrimination": prediction_b.get("discrimination", "Fair"),
            "discrimination_index": prediction_b.get("discrimination_index", 0.3),
            "feature_importance": prediction_b.get("feature_importance", prediction_a.get("feature_importance", {})),
            "confidence": prediction_b.get("confidence", prediction_a.get("confidence", 0.5)),
            "features": prediction_b.get("features", prediction_a.get("features", {})),
        }
        
        # Audit the question
        audit_flags = extractor.audit(raw_question)
        
        # Detect Bloom's level
        bloom_level = extractor.get_bloom_level(raw_question)
        
        # ─── Semantic Enrichment (if API key available) ────────────
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        semantic_scores = {}
        
        if api_key:
            # LLM rates conceptual difficulty (semantic understanding)
            llm_rating = rate_difficulty_with_llm(raw_question)
            
            # Student simulator reports confidence (inverse = difficulty proxy)
            student_sim = simulate_student_confidence(raw_question)
            
            # Blend XGBoost + LLM + Student into final difficulty estimate
            blend = blend_difficulty_scores(
                xgb_difficulty_index=ml_stats["difficulty_index"],
                llm_difficulty_score=llm_rating["llm_difficulty_score"],
                student_confidence=student_sim["student_confidence"],
                llm_available=llm_rating["available"],
                student_available=student_sim["available"],
            )
            
            # Override XGBoost-only predictions with blended result
            ml_stats["difficulty"] = blend["blended_difficulty"]
            ml_stats["difficulty_index"] = blend["blended_difficulty_index"]
            
            semantic_scores = {
                "llm_rating": llm_rating,
                "student_sim": student_sim,
                "blend": blend,
            }
        
        # Build status message
        status = (
            f"📊 **Analysis Complete**\n"
            f"- **Difficulty**: {ml_stats['difficulty']} "
            f"(Index: {ml_stats['difficulty_index']:.2f}, "
            f"Confidence: {ml_stats['confidence']:.1%})\n"
            f"- **Discrimination**: {ml_stats['discrimination']} "
            f"(Index: {ml_stats['discrimination_index']:.2f})\n"
            f"- **Bloom's Level**: {bloom_level}\n"
            f"- **Audit Warnings**: {len(audit_flags)} issue(s) found"
        )
        
        if semantic_scores:
            llm_r = semantic_scores["llm_rating"]
            stu_r = semantic_scores["student_sim"]
            blend = semantic_scores["blend"]
            status += (
                f"\n- **Semantic Scores**: XGBoost={blend['component_scores']['xgboost']:.2f} | "
                f"LLM={blend['component_scores']['llm_semantic']:.2f} | "
                f"Student confidence={stu_r['student_confidence']:.2f}"
            )
        
        # Top feature drivers
        top_features = list(ml_stats["feature_importance"].items())[:3]
        if top_features:
            drivers = ", ".join(
                f"{name} ({val:.1%})" for name, val in top_features
            )
            status += f"\n- **Key Drivers**: {drivers}"
        
        return {
            "ml_stats": ml_stats,
            "feature_audit": audit_flags,
            "current_bloom_level": bloom_level,
            "semantic_scores": semantic_scores,
            "messages": [status],
            "error": ""
        }
        
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        return {
            "ml_stats": {},
            "feature_audit": [f"❌ ERROR: {error_msg}"],
            "current_bloom_level": "Unknown",
            "messages": [f"❌ {error_msg}"],
            "error": error_msg
        }
