"""
Semantic Scoring: LLM-based difficulty rating and student simulation.

Augments XGBoost's surface-level features with semantic understanding:

1. LLMDifficultyRater  — asks the LLM to rate conceptual difficulty 1-10,
   focusing on cognitive depth (not length/LaTeX).

2. StudentSimulator    — simulates an average student answering the question
   and reports their confidence (0-1). Low confidence → harder question.

3. blend_difficulty_scores — combines XGBoost + LLM + student into a single
   blended difficulty index, with graceful fallback when LLM is unavailable.
"""
import os
import re
import json
from config.settings import GROQ_API_KEY, LLM_MODEL


# ─── Helpers ─────────────────────────────────────────────────

def _get_llm(temperature: float = 0.1, max_tokens: int = 250):
    """Get a lightweight Groq LLM instance. Returns None if no API key."""
    from langchain_groq import ChatGroq
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    return ChatGroq(
        api_key=api_key,
        model_name=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response (handles <think> tags)."""
    from utils.think_parser import separate_thinking
    clean, _ = separate_thinking(text)
    match = re.search(r'\{[^{}]+\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ─── LLM Difficulty Rater ─────────────────────────────────────

def rate_difficulty_with_llm(question_text: str) -> dict:
    """Ask the LLM to rate the CONCEPTUAL difficulty of a question (1-10).

    This captures semantic understanding that XGBoost misses — a short
    question about bijections can be harder than a lengthy LaTeX question.

    Args:
        question_text: Raw MCQ text (stem + options).

    Returns:
        {
            'llm_difficulty_score': float,  # 0.0–1.0 (normalized from 1–10)
            'llm_bloom_level': str,
            'llm_reasoning': str,
            'raw_score': float,             # original 1–10 rating
            'available': bool,
        }
    """
    llm = _get_llm(temperature=0.1, max_tokens=250)
    if llm is None:
        return {
            'llm_difficulty_score': 0.5,
            'llm_bloom_level': 'Unknown',
            'llm_reasoning': 'LLM unavailable (no API key)',
            'raw_score': 5.0,
            'available': False,
        }

    prompt = f"""Rate the CONCEPTUAL difficulty of this MCQ question on a scale of 1-10.

Focus on cognitive depth required — NOT question length, LaTeX, or formatting.
1  = Trivially easy (pure recall of a definition)
5  = Moderate (requires application of a concept to a new scenario)
10 = Expert level (requires synthesis, evaluation, or detection of subtle flaws)

Question:
{question_text}

Respond with ONLY valid JSON (no text outside the JSON):
{{"difficulty_score": <1-10>, "bloom_level": "<Remember|Understand|Apply|Analyze|Evaluate|Create>", "reasoning": "<one concise sentence>"}}"""

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        data = _extract_json(response.content)

        raw_score = float(data.get('difficulty_score', 5.0))
        raw_score = max(1.0, min(10.0, raw_score))

        return {
            'llm_difficulty_score': round((raw_score - 1.0) / 9.0, 4),  # normalize to 0–1
            'llm_bloom_level': data.get('bloom_level', 'Unknown'),
            'llm_reasoning': data.get('reasoning', ''),
            'raw_score': raw_score,
            'available': True,
        }
    except Exception as e:
        return {
            'llm_difficulty_score': 0.5,
            'llm_bloom_level': 'Unknown',
            'llm_reasoning': f'Rating error: {str(e)}',
            'raw_score': 5.0,
            'available': False,
        }


# ─── Student Simulator ────────────────────────────────────────

def simulate_student_confidence(question_text: str) -> dict:
    """Simulate an average student answering the question.

    Their reported confidence acts as an inverse proxy for difficulty:
        Low confidence  → student found it hard → harder question
        High confidence → student found it easy → easier question

    Args:
        question_text: Raw MCQ text (stem + options).

    Returns:
        {
            'student_confidence': float,  # 0.0–1.0
            'student_selected': str,      # 'A' | 'B' | 'C' | 'D'
            'student_reasoning': str,
            'available': bool,
        }
    """
    llm = _get_llm(temperature=0.3, max_tokens=200)
    if llm is None:
        return {
            'student_confidence': 0.5,
            'student_selected': 'Unknown',
            'student_reasoning': 'LLM unavailable (no API key)',
            'available': False,
        }

    prompt = f"""You are an average university student with moderate subject knowledge.
Attempt this MCQ honestly — show genuine uncertainty when you are unsure.

Question:
{question_text}

Respond with ONLY valid JSON:
{{"selected": "<A|B|C|D>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation including any doubts>"}}

confidence = 0.0 means complete guess, 1.0 means absolutely certain."""

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        data = _extract_json(response.content)

        confidence = float(data.get('confidence', 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            'student_confidence': round(confidence, 4),
            'student_selected': data.get('selected', 'Unknown'),
            'student_reasoning': data.get('reasoning', ''),
            'available': True,
        }
    except Exception as e:
        return {
            'student_confidence': 0.5,
            'student_selected': 'Unknown',
            'student_reasoning': f'Simulation error: {str(e)}',
            'available': False,
        }


# ─── Score Blender ────────────────────────────────────────────

def blend_difficulty_scores(
    xgb_difficulty_index: float,
    llm_difficulty_score: float,
    student_confidence: float,
    llm_available: bool,
    student_available: bool,
) -> dict:
    """Blend XGBoost, LLM, and student scores into a final difficulty estimate.

    Weights (when all sources are available):
        XGBoost  50% — structural/format features (fast, deterministic)
        LLM      30% — semantic/cognitive depth understanding
        Student  20% — lived difficulty proxy (1 - confidence)

    When a source is unavailable, its weight redistributes to XGBoost.

    Args:
        xgb_difficulty_index: XGBoost regression output (0–1, higher = harder).
        llm_difficulty_score: LLM rating normalized to 0–1.
        student_confidence: Student's self-reported confidence (0–1).
        llm_available: Whether the LLM rating succeeded.
        student_available: Whether the student simulation succeeded.

    Returns:
        {
            'blended_difficulty_index': float,
            'blended_difficulty': str ('Easy' | 'Medium' | 'Hard'),
            'component_scores': dict,
        }
    """
    # Student difficulty proxy: low confidence = harder question
    student_difficulty_proxy = 1.0 - student_confidence

    if llm_available and student_available:
        blended = (
            0.50 * xgb_difficulty_index +
            0.30 * llm_difficulty_score +
            0.20 * student_difficulty_proxy
        )
    elif llm_available:
        blended = 0.60 * xgb_difficulty_index + 0.40 * llm_difficulty_score
    elif student_available:
        blended = 0.70 * xgb_difficulty_index + 0.30 * student_difficulty_proxy
    else:
        blended = xgb_difficulty_index  # fall back to XGBoost only

    blended = round(max(0.0, min(1.0, blended)), 4)

    # Re-classify
    if blended < 0.35:
        difficulty_class = 'Easy'
    elif blended < 0.65:
        difficulty_class = 'Medium'
    else:
        difficulty_class = 'Hard'

    return {
        'blended_difficulty_index': blended,
        'blended_difficulty': difficulty_class,
        'component_scores': {
            'xgboost': round(xgb_difficulty_index, 4),
            'llm_semantic': round(llm_difficulty_score, 4) if llm_available else None,
            'student_proxy': round(student_difficulty_proxy, 4) if student_available else None,
        },
    }
