"""
Node 4: The Refiner

Generates three versions of the question (Easy, Medium, Hard) based on
the reasoning and improvement strategy. Validates each version using
XGBoost to ensure difficulty alignment.
"""
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ml.predictor import QuestionPredictor
from config.settings import (
    GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, 
    LLM_MAX_TOKENS, MAX_REFINEMENT_ITERATIONS
)
from utils.think_parser import separate_thinking

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = QuestionPredictor()
    return _predictor


def _get_llm():
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(
        api_key=api_key,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


REFINER_SYSTEM_PROMPT = """You are an expert MCQ question writer and assessment designer.
Your task is to generate refined versions of educational assessment questions at different difficulty levels.

Follow these rules strictly:
1. Each question must have exactly 4 options (A, B, C, D)
2. Mark the correct answer clearly
3. Each version must be self-contained (no references to other versions)
4. Distractors must target specific misconceptions
5. Use clear, concise language appropriate for the difficulty level

Format each question version EXACTLY as follows:

### EASY VERSION
**Bloom's Level**: [level]
**Stem**: [question text]
A) [option]
B) [option]  
C) [option]
D) [option]
**Correct Answer**: [letter]
**Rationale**: [why this difficulty level]

### MEDIUM VERSION
[same format]

### HARD VERSION
[same format]"""


def _build_refiner_prompt(state: dict) -> str:
    """Build the question generation prompt."""
    raw_question = state.get("raw_question", "")
    reasoning = state.get("reasoning", "")
    improvement_strategy = state.get("improvement_strategy", "")
    ml_stats = state.get("ml_stats", {})
    target_bloom = state.get("target_bloom_level", "")
    target_difficulty = state.get("target_difficulty", "")
    subject = state.get("subject", "General")
    user_request = state.get("user_request", "")
    
    prompt = f"""## Original Question
```
{raw_question}
```

## Current Analysis
- Difficulty: {ml_stats.get('difficulty', 'Unknown')}
- Discrimination: {ml_stats.get('discrimination', 'Unknown')}
- Current Bloom's Level: {state.get('current_bloom_level', 'Unknown')}

## Expert Reasoning
{reasoning[:1500]}

## Improvement Strategy
{improvement_strategy[:800]}

## Requirements
- Subject: {subject}
"""

    if target_bloom:
        prompt += f"- Target Bloom's Level: {target_bloom}\n"
    if target_difficulty:
        prompt += f"- Prioritized Difficulty: {target_difficulty}\n"
    if user_request:
        prompt += f"- User Request: {user_request}\n"

    prompt += """
## Task
Generate THREE versions of this question:
1. **EASY** — Tests recall/understanding (Bloom's: Remember/Understand)
2. **MEDIUM** — Tests application (Bloom's: Apply/Analyze)
3. **HARD** — Tests evaluation/creation (Bloom's: Evaluate/Create)

Each version must cover the same core concept but at different cognitive levels.
Ensure distractors target realistic student misconceptions.
Follow the exact format specified in the system prompt."""

    return prompt


def _parse_refined_questions(response_text: str) -> dict:
    """Parse the LLM response into structured question versions."""
    questions = {"easy": None, "medium": None, "hard": None}
    
    current_level = None
    current_content = []
    
    for line in response_text.split('\n'):
        line_lower = line.lower().strip()
        
        if 'easy version' in line_lower or '**easy' in line_lower:
            if current_level and current_content:
                questions[current_level] = '\n'.join(current_content).strip()
            current_level = "easy"
            current_content = []
        elif 'medium version' in line_lower or '**medium' in line_lower:
            if current_level and current_content:
                questions[current_level] = '\n'.join(current_content).strip()
            current_level = "medium"
            current_content = []
        elif 'hard version' in line_lower or '**hard' in line_lower:
            if current_level and current_content:
                questions[current_level] = '\n'.join(current_content).strip()
            current_level = "hard"
            current_content = []
        elif current_level:
            current_content.append(line)
    
    # Capture last section
    if current_level and current_content:
        questions[current_level] = '\n'.join(current_content).strip()
    
    # Fallback if parsing failed
    if not any(questions.values()):
        questions["medium"] = response_text.strip()
    
    return questions


def refiner_node(state: dict) -> dict:
    """Generate and validate refined question versions.
    
    Inputs from state:
        - raw_question, reasoning, improvement_strategy
        - ml_stats, target_bloom_level, target_difficulty, subject
        
    Outputs to state:
        - refined_questions: Dict of easy/medium/hard versions
        - difficulty_justification: Rationale for each version
        - validation_results: XGBoost re-validation
        - should_continue: Whether to loop back
        - iteration_count: Updated count
        - critique: Issues found during validation
        - messages: Status messages
    """
    iteration_count = state.get("iteration_count", 0)
    target_difficulty = state.get("target_difficulty", "")
    
    try:
        llm = _get_llm()
        prompt = _build_refiner_prompt(state)
        
        messages = [
            SystemMessage(content=REFINER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Strip <think> blocks before parsing question versions
        clean_response, _ = separate_thinking(response.content)
        refined = _parse_refined_questions(clean_response)
        
        # Validate each version with XGBoost
        predictor = _get_predictor()
        validation = {}
        critique_items = []
        
        for level, question_text in refined.items():
            if question_text:
                result = predictor.validate_question(
                    question_text, 
                    target_difficulty=level.capitalize(),
                    model="A"
                )
                validation[level] = {
                    "predicted_difficulty": result["predicted"],
                    "target_difficulty": result["target"],
                    "matches": result["matches"],
                    "difficulty_index": result["prediction"].get("difficulty_index", 0),
                    "confidence": result["prediction"].get("confidence", 0),
                }
                
                if not result["matches"]:
                    critique_items.append(
                        f"'{level}' version predicted as {result['predicted']} "
                        f"(target: {result['target']})"
                    )
        
        # Determine if we need to loop
        has_mismatch = len(critique_items) > 0
        should_continue = (
            has_mismatch 
            and iteration_count < MAX_REFINEMENT_ITERATIONS - 1
            and bool(target_difficulty)  # Only loop if user specified a target
        )
        
        critique = ""
        if critique_items:
            critique = "Difficulty mismatches found:\n" + "\n".join(
                f"- {item}" for item in critique_items
            )
        
        # Build justification
        justification_parts = []
        for level in ["easy", "medium", "hard"]:
            if level in validation:
                v = validation[level]
                match_icon = "✅" if v["matches"] else "⚠️"
                justification_parts.append(
                    f"**{level.capitalize()}**: {match_icon} Predicted as "
                    f"{v['predicted_difficulty']} (DI: {v['difficulty_index']:.2f}, "
                    f"Confidence: {v['confidence']:.1%})"
                )
        
        justification = "\n".join(justification_parts)
        
        # Status
        match_count = sum(1 for v in validation.values() if v.get("matches", False))
        total = len(validation)
        
        status = (
            f"✨ **Refined Questions Generated** (Iteration {iteration_count + 1})\n"
            f"- Versions created: {sum(1 for v in refined.values() if v)}\n"
            f"- XGBoost validation: {match_count}/{total} match target difficulty\n"
        )
        
        if should_continue:
            status += f"- 🔄 Looping back for refinement (mismatches found)"
        else:
            status += f"- ✅ Refinement complete"
        
        return {
            "refined_questions": refined,
            "difficulty_justification": justification,
            "validation_results": validation,
            "should_continue": should_continue,
            "iteration_count": iteration_count + 1,
            "critique": critique,
            "messages": [status],
        }
        
    except Exception as e:
        error_msg = f"Refiner error: {str(e)}"
        return {
            "refined_questions": {"easy": None, "medium": None, "hard": None},
            "difficulty_justification": f"Error during refinement: {error_msg}",
            "validation_results": {},
            "should_continue": False,
            "iteration_count": iteration_count + 1,
            "critique": error_msg,
            "messages": [f"⚠️ {error_msg}"],
        }
