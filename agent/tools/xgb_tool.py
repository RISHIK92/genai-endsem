"""
XGBoost as a LangChain-compatible Tool.

Wraps the QuestionPredictor so the LLM agent can call it as a tool
to validate generated questions.
"""
from langchain_core.tools import tool
from ml.feature_extractor import FeatureExtractor
from ml.predictor import QuestionPredictor

_predictor = None
_extractor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = QuestionPredictor()
    return _predictor


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


@tool
def predict_question_difficulty(question_text: str) -> str:
    """Predict the difficulty and discrimination of an MCQ question
    using the trained XGBoost models.
    
    Args:
        question_text: The full MCQ text including stem and all options.
    
    Returns:
        A formatted string with difficulty, discrimination, and key feature drivers.
    """
    predictor = _get_predictor()
    result = predictor.predict(question_text, model="B")
    
    output = (
        f"Difficulty: {result['difficulty']} (Index: {result['difficulty_index']:.2f})\n"
        f"Discrimination: {result['discrimination']} (Index: {result['discrimination_index']:.2f})\n"
        f"Confidence: {result['confidence']:.1%}\n"
        f"Class Probabilities: {result['difficulty_probabilities']}\n"
    )
    
    importance = result.get("feature_importance", {})
    if importance:
        output += "Top Feature Drivers:\n"
        for name, val in list(importance.items())[:5]:
            output += f"  - {name}: {val:.1%}\n"
    
    return output


@tool
def validate_question_difficulty(question_text: str, target_difficulty: str) -> str:
    """Validate whether a generated MCQ matches the target difficulty level.
    
    Args:
        question_text: The full MCQ text including stem and all options.
        target_difficulty: The intended difficulty level (Easy, Medium, or Hard).
        
    Returns:
        Validation result indicating match/mismatch and predicted values.
    """
    predictor = _get_predictor()
    result = predictor.validate_question(question_text, target_difficulty, model="A")
    
    match_str = "✅ MATCH" if result["matches"] else "❌ MISMATCH"
    return (
        f"{match_str}\n"
        f"Target: {result['target']}\n"
        f"Predicted: {result['predicted']}\n"
        f"Difficulty Index: {result['prediction']['difficulty_index']:.2f}\n"
        f"Confidence: {result['prediction']['confidence']:.1%}"
    )
