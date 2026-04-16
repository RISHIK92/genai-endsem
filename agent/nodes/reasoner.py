"""
Node 3: The Reasoner

Uses LLM (Groq) to analyze the question by combining ML predictions
with pedagogical research context. Produces reasoning and improvement strategy.
"""
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from utils.think_parser import separate_thinking


def _get_llm():
    """Initialize Groq LLM."""
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        )
    
    return ChatGroq(
        api_key=api_key,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


REASONER_SYSTEM_PROMPT = """You are a Senior Examiner and Assessment Specialist with deep expertise in:
- Bloom's Taxonomy and cognitive levels
- Item Response Theory (IRT) and psychometrics
- MCQ design best practices
- Subject-specific pedagogy

Your role is to analyze educational assessment questions using machine learning predictions
and pedagogical research. You provide expert reasoning about WHY a question has certain
statistical properties and HOW to improve it.

Always be specific, actionable, and grounded in educational research.
Refer to the ML statistics and pedagogical context provided."""


def _build_reasoner_prompt(state: dict) -> str:
    """Build the detailed analysis prompt for the LLM."""
    ml_stats = state.get("ml_stats", {})
    raw_question = state.get("raw_question", "")
    user_request = state.get("user_request", "")
    pedagogy_context = state.get("pedagogy_context", "")
    feature_audit = state.get("feature_audit", [])
    current_bloom = state.get("current_bloom_level", "Unknown")
    target_bloom = state.get("target_bloom_level", "")
    target_difficulty = state.get("target_difficulty", "")
    iteration_count = state.get("iteration_count", 0)
    
    # Format feature importance
    importance = ml_stats.get("feature_importance", {})
    importance_str = "\n".join(
        f"  - {name}: {val:.1%}" for name, val in list(importance.items())[:5]
    ) or "  (not available)"
    
    # Format audit warnings
    audit_str = "\n".join(f"  {flag}" for flag in feature_audit) or "  None"
    
    # Format difficulty probabilities
    probs = ml_stats.get("difficulty_probabilities", {})
    probs_str = ", ".join(f"{k}: {v:.1%}" for k, v in probs.items()) or "N/A"
    
    prompt = f"""## Question Under Review
```
{raw_question}
```

## ML Analysis Results
- **Predicted Difficulty**: {ml_stats.get('difficulty', 'Unknown')} (Index: {ml_stats.get('difficulty_index', 'N/A')})
- **Class Probabilities**: {probs_str}
- **Confidence**: {ml_stats.get('confidence', 'N/A')}
- **Discrimination**: {ml_stats.get('discrimination', 'Unknown')} (Index: {ml_stats.get('discrimination_index', 'N/A')})
- **Current Bloom's Level**: {current_bloom}

### Key Feature Drivers
{importance_str}

### Audit Warnings
{audit_str}

## Pedagogical Research Context
{pedagogy_context}

## User Request
{user_request if user_request else 'Analyze this question and suggest improvements.'}
"""
    
    if target_bloom:
        prompt += f"\n## Target Bloom's Level: {target_bloom}"
    if target_difficulty:
        prompt += f"\n## Target Difficulty: {target_difficulty}"
    
    if iteration_count > 0:
        critique = state.get("critique", "")
        prompt += f"""

## Iteration {iteration_count} — Previous Critique
{critique}
Please address the issues identified in the previous iteration.
"""
    
    prompt += """

## Your Task
Please provide:

1. **Root Cause Analysis**: Explain WHY this question has its predicted difficulty and discrimination.
   Focus on specific features (stem length, LaTeX, distractors, Bloom's level).

2. **Bloom's Taxonomy Assessment**: Identify the exact Bloom's level and cognitive skills being tested.
   If the current level is misaligned with the target, explain the gap.

3. **Improvement Strategy**: Provide a concrete, step-by-step plan to improve the question.
   Be specific about what to change in the stem, options, and distractors.

4. **Key Modifications**: List the 3-5 most impactful changes to make.

Format your response clearly with headers for each section."""
    
    return prompt


def reasoner_node(state: dict) -> dict:
    """Analyze the question using LLM reasoning.
    
    Inputs from state:
        - raw_question, ml_stats, feature_audit, pedagogy_context
        - user_request, target_bloom_level, target_difficulty
        
    Outputs to state:
        - reasoning: Full LLM analysis
        - improvement_strategy: Extracted strategy
        - messages: Status message
    """
    try:
        llm = _get_llm()
        prompt = _build_reasoner_prompt(state)
        
        messages = [
            SystemMessage(content=REASONER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        raw_reasoning = response.content
        
        # Separate <think> blocks from the response
        reasoning, thinking_content = separate_thinking(raw_reasoning)
        
        # Extract improvement strategy section
        strategy = ""
        in_strategy = False
        for line in reasoning.split('\n'):
            if 'improvement strategy' in line.lower() or 'key modifications' in line.lower():
                in_strategy = True
                strategy += line + "\n"
            elif in_strategy:
                if line.startswith('## ') or line.startswith('# '):
                    break
                strategy += line + "\n"
        
        if not strategy:
            strategy = reasoning  # Fallback to full reasoning
        
        status = "🧠 **Reasoning Complete** — Analysis and improvement strategy generated."
        
        return {
            "reasoning": reasoning,
            "improvement_strategy": strategy.strip(),
            "thinking": thinking_content,
            "messages": [status],
        }
        
    except Exception as e:
        error_msg = f"Reasoner error: {str(e)}"
        
        # Provide a synthetic reasoning fallback
        ml_stats = state.get("ml_stats", {})
        fallback = _generate_fallback_reasoning(state)
        
        return {
            "reasoning": fallback,
            "improvement_strategy": "Review the ML analysis and apply standard MCQ improvement guidelines.",
            "messages": [f"⚠️ {error_msg} — using fallback reasoning"],
        }


def _generate_fallback_reasoning(state: dict) -> str:
    """Generate fallback reasoning when LLM is unavailable."""
    ml_stats = state.get("ml_stats", {})
    feature_audit = state.get("feature_audit", [])
    bloom = state.get("current_bloom_level", "Unknown")
    
    difficulty = ml_stats.get("difficulty", "Unknown")
    discrimination = ml_stats.get("discrimination", "Unknown")
    
    reasoning = f"""## Root Cause Analysis (Fallback Mode)

The question has been predicted as **{difficulty}** difficulty with **{discrimination}** discrimination.

### Key Issues Identified:
"""
    
    for flag in feature_audit:
        reasoning += f"- {flag}\n"
    
    reasoning += f"""
### Bloom's Taxonomy Assessment
Current level detected: **{bloom}**

### Improvement Strategy
1. Review and address each audit warning above
2. Ensure alignment between question content and intended difficulty
3. Improve distractor quality to enhance discrimination
4. Consider adjusting the cognitive level as needed

*Note: Detailed LLM-powered reasoning unavailable. Please check your GROQ_API_KEY configuration.*
"""
    
    return reasoning
