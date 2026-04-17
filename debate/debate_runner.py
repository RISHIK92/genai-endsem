"""
Multi-Agent Debate: The Synthetic Student Batch

Three agents debate to produce an optimally refined question:
- Agent A (Professor): Designs/refines questions
- Agent B (ML Predictor): Validates difficulty with XGBoost
- Agent C (Student Persona): Simulates student misconceptions

Orchestrated by DebateRunner.
"""
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ml.predictor import QuestionPredictor
from ml.feature_extractor import FeatureExtractor
from config.settings import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from utils.think_parser import separate_thinking


def _get_llm(temperature=None):
    """Get Groq LLM instance."""
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(
        api_key=api_key,
        model_name=LLM_MODEL,
        temperature=temperature or LLM_TEMPERATURE,
        max_tokens=1500,
    )


class ProfessorAgent:
    """Agent A: The Professor — designs and refines questions."""
    
    SYSTEM_PROMPT = """You are a university professor and expert question designer.
Your task is to create or refine MCQ questions that precisely target specific 
Bloom's Taxonomy levels and difficulty targets.

Rules:
- Always provide exactly 4 options (A-D)
- Distractors must target specific misconceptions
- The stem must be clear and concise
- Mark the correct answer
- Explain your design choices briefly

Format your question as:
STEM: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
CORRECT: [letter]
DESIGN NOTES: [brief explanation]"""

    def design(self, topic: str, bloom_level: str, 
               target_difficulty: str, feedback: str = "") -> str:
        """Design or refine a question."""
        llm = _get_llm(temperature=0.8)
        
        prompt = f"""Design an MCQ question with these specifications:
- Topic: {topic}
- Bloom's Level: {bloom_level}
- Target Difficulty: {target_difficulty}
"""
        if feedback:
            prompt += f"\n## Feedback from previous round:\n{feedback}\n\nAddress this feedback in your revision."
        
        response = llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        clean, thinking = separate_thinking(response.content)
        return clean, thinking

    def refine(self, original_question: str, feedback: str,
               target_difficulty: str) -> str:
        """Refine a question based on feedback."""
        llm = _get_llm(temperature=0.7)
        
        prompt = f"""Refine this question based on the feedback:

## Original Question
{original_question}

## Feedback
{feedback}

## Target
- Difficulty: {target_difficulty}

Provide the refined version in the standard format."""
        
        response = llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        clean, thinking = separate_thinking(response.content)
        return clean, thinking


class PredictorAgent:
    """Agent B: The ML Predictor — validates difficulty with XGBoost."""
    
    def __init__(self):
        self.predictor = QuestionPredictor()
        self.extractor = FeatureExtractor()
    
    def validate(self, question_text: str, target_difficulty: str) -> dict:
        """Validate a question's difficulty."""
        prediction = self.predictor.predict(question_text, model="B")
        audit = self.extractor.audit(question_text)
        bloom = self.extractor.get_bloom_level(question_text)
        
        matches = prediction["difficulty"] == target_difficulty
        
        feedback = f"ML Prediction: {prediction['difficulty']} "
        feedback += f"(Target: {target_difficulty}) — "
        feedback += "✅ MATCH" if matches else "❌ MISMATCH"
        feedback += f"\nDifficulty Index: {prediction['difficulty_index']:.2f}"
        feedback += f"\nDiscrimination: {prediction['discrimination']}"
        feedback += f"\nBloom's Level: {bloom}"
        
        if prediction.get("feature_importance"):
            top = list(prediction["feature_importance"].items())[:3]
            feedback += "\nKey Drivers: " + ", ".join(
                f"{k} ({v:.1%})" for k, v in top
            )
        
        if audit:
            feedback += "\nAudit Issues: " + "; ".join(audit[:3])
        
        return {
            "feedback": feedback,
            "matches": matches,
            "prediction": prediction,
            "audit": audit,
            "bloom_level": bloom,
        }


class StudentAgent:
    """Agent C: The Student Persona — simulates student misconceptions."""
    
    SYSTEM_PROMPT = """You are simulating a student attempting an MCQ question.
    
Based on the question and your assigned profile, you must:
1. Try to answer the question (show your reasoning)
2. Identify which option you'd select and why
3. Note any confusing elements
4. Point out if any distractors seem too obvious or too tricky

Be realistic — make the kinds of mistakes a real student would make.
Don't just pick the correct answer; show genuine reasoning including potential errors."""

    def attempt(self, question_text: str, student_type: str = "average",
                misconception_hints: list = None) -> str:
        """Simulate a student attempting the question."""
        llm = _get_llm(temperature=0.9)
        
        profile_descriptions = {
            "high_achieving": "You are a high-achieving student who usually gets A grades. You read carefully but sometimes overthink simple questions.",
            "average": "You are an average student with moderate understanding. You sometimes confuse similar concepts and may rush through questions.",
            "struggling": "You are a struggling student who often has misconceptions. You tend to be attracted to familiar-looking but incorrect options."
        }
        
        profile = profile_descriptions.get(student_type, profile_descriptions["average"])
        
        prompt = f"""## Your Student Profile
{profile}

## Question to Attempt
{question_text}
"""
        if misconception_hints:
            prompt += f"\n## Known Misconceptions in This Topic\n"
            for hint in misconception_hints[:3]:
                prompt += f"- {hint}\n"
        
        prompt += """
## Your Task
1. Read the question carefully
2. Show your thinking process (including any confusion)
3. Select your answer and explain why
4. Rate the question's clarity (1-5)
5. Note any confusing or unfair elements"""
        
        response = llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        clean, thinking = separate_thinking(response.content)
        return clean, thinking


class DebateRunner:
    """Orchestrates the multi-agent debate to produce optimally refined questions."""
    
    def __init__(self):
        self.professor = ProfessorAgent()
        self.predictor = PredictorAgent()
        self.student = StudentAgent()
    
    def run_debate(self, topic: str, bloom_level: str,
                   target_difficulty: str, max_rounds: int = 3,
                   subject: str = "General") -> dict:
        """Run a multi-agent debate to produce a refined question.
        
        Args:
            topic: The topic/concept to test
            bloom_level: Target Bloom's Taxonomy level
            target_difficulty: Target difficulty (Easy/Medium/Hard)
            max_rounds: Maximum debate rounds
            subject: Subject area
            
        Returns:
            {
                'final_question': str,
                'debate_transcript': list[dict],
                'difficulty_report': dict,
                'consensus_reached': bool,
                'rounds_taken': int
            }
        """
        transcript = []
        current_question = None
        consensus = False
        
        for round_num in range(1, max_rounds + 1):
            round_entry = {"round": round_num, "exchanges": []}
            
            # ─── Professor designs/refines ──────────────────────
            if current_question is None:
                professor_output, professor_thinking = self.professor.design(
                    topic=topic,
                    bloom_level=bloom_level,
                    target_difficulty=target_difficulty
                )
                round_entry["exchanges"].append({
                    "agent": "Professor",
                    "action": "Initial Design",
                    "output": professor_output,
                    "thinking": professor_thinking
                })
            else:
                # Compile feedback from previous round
                feedback = "\n".join(
                    f"[{e['agent']}]: {e['output'][:300]}" 
                    for e in transcript[-1]["exchanges"]
                    if e["agent"] != "Professor"
                )
                professor_output, professor_thinking = self.professor.refine(
                    original_question=current_question,
                    feedback=feedback,
                    target_difficulty=target_difficulty
                )
                round_entry["exchanges"].append({
                    "agent": "Professor",
                    "action": "Refinement",
                    "output": professor_output,
                    "thinking": professor_thinking
                })
            
            current_question = professor_output
            
            # ─── ML Predictor validates ────────────────────────
            validation = self.predictor.validate(
                current_question, target_difficulty
            )
            round_entry["exchanges"].append({
                "agent": "ML Predictor",
                "action": "Validation",
                "output": validation["feedback"]
            })
            
            # ─── Student attempts ──────────────────────────────
            student_types = ["average", "struggling", "high_achieving"]
            student_type = student_types[min(round_num - 1, len(student_types) - 1)]
            
            student_output, student_thinking = self.student.attempt(
                current_question,
                student_type=student_type
            )
            round_entry["exchanges"].append({
                "agent": f"Student ({student_type})",
                "action": "Attempt",
                "output": student_output,
                "thinking": student_thinking
            })
            
            transcript.append(round_entry)
            
            # Check for consensus
            if validation["matches"]:
                consensus = True
                break
        
        # Final validation
        final_validation = self.predictor.validate(
            current_question, target_difficulty
        )
        
        return {
            "final_question": current_question,
            "debate_transcript": transcript,
            "difficulty_report": final_validation,
            "consensus_reached": consensus,
            "rounds_taken": len(transcript),
        }
