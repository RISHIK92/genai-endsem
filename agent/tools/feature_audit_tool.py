"""
Feature Audit Tool for LangChain.

Wraps the feature extractor's audit mode as a tool that the agent
can use to flag issues with LLM-generated questions.
"""
from langchain_core.tools import tool
from ml.feature_extractor import FeatureExtractor

_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


@tool
def audit_question_features(question_text: str) -> str:
    """Audit an MCQ question for potential quality issues.
    
    Checks for problems like excessive stem length, high LaTeX density,
    poor readability, distractor quality issues, and Bloom's level concerns.
    
    Args:
        question_text: The full MCQ text including stem and all options.
        
    Returns:
        A list of warnings and suggestions for improvement.
    """
    extractor = _get_extractor()
    warnings = extractor.audit(question_text)
    
    if not warnings:
        return "✅ No issues found. The question passes all quality checks."
    
    output = f"Found {len(warnings)} issue(s):\n\n"
    for warning in warnings:
        output += f"{warning}\n"
    
    return output


@tool
def get_bloom_level(question_text: str) -> str:
    """Estimate the Bloom's Taxonomy cognitive level of an MCQ question.
    
    Args:
        question_text: The full MCQ text.
        
    Returns:
        The estimated Bloom's level and explanation.
    """
    extractor = _get_extractor()
    level = extractor.get_bloom_level(question_text)
    
    level_descriptions = {
        "Remember": "Recall of facts and basic concepts (lowest cognitive level)",
        "Understand": "Explaining ideas or concepts",
        "Apply": "Using information in new situations",
        "Analyze": "Drawing connections among ideas, breaking into parts",
        "Evaluate": "Justifying a decision or course of action",
        "Create": "Producing new or original work (highest cognitive level)"
    }
    
    return (
        f"Bloom's Level: {level}\n"
        f"Description: {level_descriptions.get(level, 'Unknown')}"
    )
