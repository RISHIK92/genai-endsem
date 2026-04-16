"""
Feature Extractor for MCQ Questions.

Extracts a comprehensive feature vector from raw question text.
Model A uses 25 text-based features (matching the trained model's expected feature names).
Model B uses 30 all-features (matching the trained model's expected feature names).
"""
import re
import math
import numpy as np
from typing import Optional
from config.settings import (
    MAX_STEM_LENGTH, MAX_LATEX_DENSITY, MAX_WORD_COUNT,
    MIN_OPTION_COUNT, MIN_READABILITY_SCORE
)


class FeatureExtractor:
    """Extracts features from MCQ question text for XGBoost prediction.
    
    Feature names MUST match the ones used during model training.
    """

    # ─── Model A expected features (25 total) ────────────────
    # From the trained model: text_length, word_count, sentence_count,
    # avg_word_length, latex_command_count, has_latex, latex_density,
    # math_operator_count, number_count, vocab_richness, text_complexity_score,
    # answer_a_length, answer_b_length, answer_c_length, answer_d_length,
    # avg_answer_length, answer_length_variance, has_advanced_terms,
    # has_algebra_terms, has_geometry_terms, has_stats_terms,
    # num_misconceptions, has_misconception, subject_difficulty_tier,
    # construct_frequency

    # ─── Model B expected features (30 total) ────────────────
    # Same as Model A + 5 additional features

    # Bloom's Taxonomy keyword mapping
    BLOOM_KEYWORDS = {
        "Remember": ["define", "list", "state", "identify", "name", "recall", "recognize", "label", "match", "memorize"],
        "Understand": ["explain", "describe", "summarize", "interpret", "classify", "compare", "discuss", "distinguish", "predict"],
        "Apply": ["apply", "solve", "use", "demonstrate", "calculate", "compute", "implement", "execute", "determine"],
        "Analyze": ["analyze", "examine", "differentiate", "organize", "deconstruct", "attribute", "investigate", "contrast"],
        "Evaluate": ["evaluate", "justify", "assess", "critique", "judge", "argue", "defend", "support", "recommend"],
        "Create": ["create", "design", "construct", "develop", "formulate", "propose", "devise", "compose", "invent"]
    }

    # Negation words that increase difficulty
    NEGATION_WORDS = ["not", "except", "least", "never", "neither", "nor", "without", "incorrect", "false", "wrong"]

    # Common misconception trigger patterns
    MISCONCEPTION_PATTERNS = [
        r"common\s+(?:error|mistake|misconception)",
        r"students?\s+(?:often|frequently|commonly)\s+(?:confuse|mistake)",
        r"(?:which|what)\s+is\s+(?:not|incorrect|wrong|false)",
        r"all\s+(?:of\s+the\s+)?(?:above|following)\s+except",
        r"none\s+of\s+the\s+(?:above|following)",
    ]

    # Advanced term lists
    ADVANCED_TERMS = [
        "theorem", "corollary", "lemma", "proof", "hypothesis", "algorithm",
        "complexity", "recursion", "derivative", "integral", "eigenvalue",
        "determinant", "isomorphism", "homeomorphism", "topology"
    ]
    ALGEBRA_TERMS = [
        "equation", "variable", "polynomial", "factor", "root", "matrix",
        "vector", "linear", "quadratic", "coefficient", "expression"
    ]
    GEOMETRY_TERMS = [
        "triangle", "circle", "angle", "polygon", "area", "perimeter",
        "volume", "radius", "diameter", "hypotenuse", "congruent", "similar"
    ]
    STATS_TERMS = [
        "mean", "median", "mode", "variance", "deviation", "probability",
        "distribution", "correlation", "regression", "hypothesis test",
        "sample", "population", "confidence interval"
    ]

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def parse_question(self, question_text: str) -> dict:
        """Parse a raw question string into stem and options.
        
        Supports formats:
        - Numbered options: 1) ..., 2) ..., etc.
        - Lettered options: A) ..., B) ..., a. ..., b. ..., (A), (a)
        - Markdown: - option text
        """
        lines = question_text.strip().split('\n')
        stem_lines = []
        options = []
        
        # Pattern to match option lines
        option_pattern = re.compile(
            r'^\s*(?:'
            r'[A-Ea-e][\)\.]\s*'    # A) or a. format
            r'|\([A-Ea-e]\)\s*'      # (A) format
            r'|[1-5][\)\.]\s*'       # 1) or 1. format
            r'|-\s+'                  # - bullet format
            r')'
        )
        
        in_options = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if option_pattern.match(line):
                in_options = True
                clean = option_pattern.sub('', line).strip()
                if clean:
                    options.append(clean)
            elif not in_options:
                stem_lines.append(line)
            else:
                if options:
                    options[-1] += ' ' + line
                else:
                    stem_lines.append(line)
        
        stem = ' '.join(stem_lines)
        return {"stem": stem, "options": options}

    def _count_words(self, text: str) -> int:
        """Count words in text, excluding LaTeX commands."""
        clean = re.sub(r'\$[^$]+\$', '', text)
        clean = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', clean)
        clean = re.sub(r'[^\w\s]', ' ', clean)
        return len(clean.split())

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return max(len([s for s in sentences if s.strip()]), 1)

    def _avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = [w for w in text.split() if w.isalpha()]
        if not words:
            return 0.0
        return round(np.mean([len(w) for w in words]), 4)

    def _count_latex_commands(self, text: str) -> int:
        """Count LaTeX commands (\\command patterns)."""
        return len(re.findall(r'\\[a-zA-Z]+', text))

    def _has_latex(self, text: str) -> int:
        """Check if text contains any LaTeX."""
        return int(bool(re.search(r'\$[^$]+\$|\\[a-zA-Z]+', text)))

    def _calc_latex_density(self, text: str) -> float:
        """Calculate ratio of LaTeX content to total content."""
        total_len = len(text)
        if total_len == 0:
            return 0.0
        
        latex_patterns = [
            r'\$[^$]+\$',
            r'\$\$[^$]+\$\$',
            r'\\[a-zA-Z]+',
            r'\\(?:frac|sqrt|sum|int|prod)',
            r'\{[^}]*\}',
        ]
        
        latex_chars = 0
        for pattern in latex_patterns:
            for match in re.finditer(pattern, text):
                latex_chars += len(match.group())
        
        return min(round(latex_chars / total_len, 4), 1.0)

    def _count_math_operators(self, text: str) -> int:
        """Count mathematical operators."""
        operators = ['+', '-', '*', '/', '=', '<', '>', '≤', '≥', '≠', '±', '∑', '∫', '∂']
        return sum(text.count(op) for op in operators)

    def _count_numbers(self, text: str) -> int:
        """Count numeric literals in text."""
        return len(re.findall(r'\b\d+(?:\.\d+)?\b', text))

    def _vocab_richness(self, text: str) -> float:
        """Calculate vocabulary richness (type-token ratio)."""
        words = text.lower().split()
        if not words:
            return 0.0
        return round(len(set(words)) / len(words), 4)

    def _text_complexity_score(self, text: str) -> float:
        """Calculate a composite text complexity score.
        
        Combines multiple signals: sentence length, word length,
        vocabulary richness, and mathematical content.
        """
        word_count = self._count_words(text)
        sentence_count = self._count_sentences(text)
        avg_wl = self._avg_word_length(text)
        vocab_rich = self._vocab_richness(text)
        latex_dens = self._calc_latex_density(text)
        
        # Weighted combination
        score = (
            (word_count / sentence_count) * 0.3 +  # Avg sentence length
            avg_wl * 0.2 +                           # Word complexity
            (1 - vocab_rich) * 0.2 +                 # Repetitiveness
            latex_dens * 10 * 0.3                     # Math complexity
        )
        return round(score, 4)

    def _has_terms(self, text: str, term_list: list) -> int:
        """Check if text contains any terms from a list."""
        text_lower = text.lower()
        return int(any(term in text_lower for term in term_list))

    def _count_misconceptions(self, text: str) -> int:
        """Count misconception-related patterns."""
        count = 0
        for pattern in self.MISCONCEPTION_PATTERNS:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _has_misconception(self, text: str) -> int:
        """Check if text has any misconception patterns."""
        return int(self._count_misconceptions(text) > 0)

    def _subject_difficulty_tier(self, text: str) -> int:
        """Estimate difficulty tier (0=basic, 1=intermediate, 2=advanced)."""
        text_lower = text.lower()
        
        advanced_count = sum(1 for t in self.ADVANCED_TERMS if t in text_lower)
        if advanced_count >= 2:
            return 2
        elif advanced_count >= 1 or self._has_latex(text):
            return 1
        return 0

    def _construct_frequency(self, text: str) -> float:
        """Estimate how common/standard this construct is.
        
        Higher values = more common/standard question patterns.
        """
        text_lower = text.lower()
        standard_patterns = [
            r"what is", r"which of", r"select the", r"choose the",
            r"find the", r"calculate", r"determine", r"identify"
        ]
        
        count = sum(1 for p in standard_patterns if re.search(p, text_lower))
        return round(count / len(standard_patterns), 4)

    def _detect_negation(self, text: str) -> bool:
        """Check if stem contains negation words."""
        text_lower = text.lower()
        return any(word in text_lower.split() for word in self.NEGATION_WORDS)

    def _estimate_bloom_level(self, text: str) -> str:
        """Estimate Bloom's Taxonomy level from keywords."""
        text_lower = text.lower()
        scores = {}
        for level, keywords in self.BLOOM_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[level] = score
        
        if max(scores.values()) == 0:
            return "Remember"
        return max(scores, key=scores.get)

    def _calc_readability(self, text: str) -> float:
        """Calculate Flesch-Kincaid readability score."""
        clean = re.sub(r'\$[^$]+\$', 'FORMULA', text)
        clean = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', 'FORMULA', clean)
        
        sentences = re.split(r'[.!?]+', clean)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = clean.split()
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)
        
        def count_syllables(word):
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            if len(word) <= 3:
                return 1
            count = len(re.findall(r'[aeiouy]+', word))
            if word.endswith('e'):
                count -= 1
            return max(count, 1)
        
        num_syllables = sum(count_syllables(w) for w in words)
        score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        return round(max(min(score, 100.0), 0.0), 2)

    def _calc_option_similarity(self, options: list) -> float:
        """Calculate average pairwise similarity between options."""
        if len(options) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(options)):
            for j in range(i + 1, len(options)):
                words_i = set(options[i].lower().split())
                words_j = set(options[j].lower().split())
                if not words_i or not words_j:
                    continue
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                similarities.append(intersection / union if union > 0 else 0.0)
        
        return round(np.mean(similarities) if similarities else 0.0, 4)

    def _has_image_reference(self, text: str) -> bool:
        """Check if the question references an image or figure."""
        patterns = [
            r'figure\s*\d*', r'fig\.\s*\d*', r'diagram',
            r'image', r'graph', r'chart', r'table\s*\d*',
            r'refer\s+to\s+the', r'shown\s+(?:in|below|above)',
            r'illustration', r'picture'
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def extract_features(self, question_text: str, model: str = "A") -> dict:
        """Extract feature vector matching the trained model's expected feature names.
        
        Args:
            question_text: The full MCQ text (stem + options).
            model: "A" for text-only (25 features) or "B" for all (30 features).
            
        Returns:
            Dictionary of feature name → value, with names matching the trained model.
        """
        parsed = self.parse_question(question_text)
        stem = parsed["stem"]
        options = parsed["options"]
        full_text = question_text

        # Pad options to 4 (A, B, C, D)
        while len(options) < 4:
            options.append("")

        # ─── Model A: 25 text-based features ────────────────
        # Feature names MUST match exactly what the model expects
        features = {
            "text_length": len(full_text),
            "word_count": self._count_words(full_text),
            "sentence_count": self._count_sentences(full_text),
            "avg_word_length": self._avg_word_length(full_text),
            "latex_command_count": self._count_latex_commands(full_text),
            "has_latex": self._has_latex(full_text),
            "latex_density": self._calc_latex_density(full_text),
            "math_operator_count": self._count_math_operators(full_text),
            "number_count": self._count_numbers(full_text),
            "vocab_richness": self._vocab_richness(full_text),
            "text_complexity_score": self._text_complexity_score(full_text),
            "answer_a_length": len(options[0]),
            "answer_b_length": len(options[1]),
            "answer_c_length": len(options[2]),
            "answer_d_length": len(options[3]),
            "avg_answer_length": round(np.mean([len(o) for o in options[:4]]), 4),
            "answer_length_variance": round(np.var([len(o) for o in options[:4]]), 4),
            "has_advanced_terms": self._has_terms(full_text, self.ADVANCED_TERMS),
            "has_algebra_terms": self._has_terms(full_text, self.ALGEBRA_TERMS),
            "has_geometry_terms": self._has_terms(full_text, self.GEOMETRY_TERMS),
            "has_stats_terms": self._has_terms(full_text, self.STATS_TERMS),
            "num_misconceptions": self._count_misconceptions(full_text),
            "has_misconception": self._has_misconception(full_text),
            "subject_difficulty_tier": self._subject_difficulty_tier(full_text),
            "construct_frequency": self._construct_frequency(full_text),
        }

        if model == "B":
            # ─── Model B: Additional 5 statistical features (30 total) ──
            # These are response-data features from the training set.
            # Since we don't have real student response data at inference time,
            # we estimate them from text complexity heuristics.
            complexity = features["text_complexity_score"]
            latex_dens = features["latex_density"]
            word_cnt = features["word_count"]
            
            # Estimate avg response time (seconds) from question complexity
            # More complex questions → longer response time
            est_response_time = 30.0 + complexity * 5.0 + latex_dens * 60.0 + word_cnt * 0.5
            
            features.update({
                "avg_response_time_sec": round(est_response_time, 4),
                "std_response_time_sec": round(est_response_time * 0.4, 4),  # ~40% CV
                "discrimination_index": 0.3,   # Neutral default
                "point_biserial_corr": 0.25,   # Neutral default
                "irt_a_param": 1.0,            # Neutral IRT discrimination
            })

        return features

    def extract_feature_vector(self, question_text: str, model: str = "A") -> np.ndarray:
        """Extract features as a numpy array (ordered for XGBoost input)."""
        features = self.extract_features(question_text, model=model)
        return np.array(list(features.values()), dtype=np.float32)

    def get_feature_names(self, model: str = "A") -> list:
        """Get ordered list of feature names for a model."""
        dummy_features = self.extract_features(
            "What is X?\nA) 1\nB) 2\nC) 3\nD) 4", model=model
        )
        return list(dummy_features.keys())

    def extract_features_extended(self, question_text: str) -> dict:
        """Extract ALL features (for audit & display, not model input).
        
        Returns a superset of features with human-readable names.
        """
        parsed = self.parse_question(question_text)
        stem = parsed["stem"]
        options = parsed["options"]
        full_text = question_text

        while len(options) < 4:
            options.append("")

        return {
            # Basic text features
            "text_length": len(full_text),
            "word_count": self._count_words(full_text),
            "sentence_count": self._count_sentences(full_text),
            "stem_length": len(stem),
            "avg_word_length": self._avg_word_length(full_text),
            
            # Math/LaTeX features
            "latex_command_count": self._count_latex_commands(full_text),
            "has_latex": self._has_latex(full_text),
            "latex_density": self._calc_latex_density(full_text),
            "math_operator_count": self._count_math_operators(full_text),
            "number_count": self._count_numbers(full_text),
            
            # Complexity
            "vocab_richness": self._vocab_richness(full_text),
            "text_complexity_score": self._text_complexity_score(full_text),
            "readability_score": self._calc_readability(full_text),
            
            # Option features
            "option_count": len(parsed["options"]),
            "answer_a_length": len(options[0]),
            "answer_b_length": len(options[1]),
            "answer_c_length": len(options[2]),
            "answer_d_length": len(options[3]),
            "avg_answer_length": round(np.mean([len(o) for o in options[:4]]), 4),
            "answer_length_variance": round(np.var([len(o) for o in options[:4]]), 4),
            "distractor_similarity": self._calc_option_similarity(options[:4]),
            
            # Domain terms
            "has_advanced_terms": self._has_terms(full_text, self.ADVANCED_TERMS),
            "has_algebra_terms": self._has_terms(full_text, self.ALGEBRA_TERMS),
            "has_geometry_terms": self._has_terms(full_text, self.GEOMETRY_TERMS),
            "has_stats_terms": self._has_terms(full_text, self.STATS_TERMS),
            
            # Misconceptions
            "num_misconceptions": self._count_misconceptions(full_text),
            "has_misconception": self._has_misconception(full_text),
            
            # Structure
            "subject_difficulty_tier": self._subject_difficulty_tier(full_text),
            "construct_frequency": self._construct_frequency(full_text),
            "has_negation": int(self._detect_negation(stem)),
            "has_image_reference": int(self._has_image_reference(full_text)),
            
            # Bloom's
            "bloom_level": self._estimate_bloom_level(full_text),
        }

    def audit(self, question_text: str) -> list:
        """Audit a question for potential issues."""
        features = self.extract_features_extended(question_text)
        parsed = self.parse_question(question_text)
        bloom_level = self._estimate_bloom_level(question_text)
        warnings = []

        # Length checks
        if features["stem_length"] > MAX_STEM_LENGTH:
            warnings.append(
                f"⚠️ WARN: Stem length {features['stem_length']} chars exceeds "
                f"recommended {MAX_STEM_LENGTH}"
            )
        
        if features["word_count"] > MAX_WORD_COUNT:
            warnings.append(
                f"⚠️ WARN: Word count {features['word_count']} exceeds "
                f"recommended {MAX_WORD_COUNT}"
            )

        # LaTeX complexity
        if features["latex_density"] > MAX_LATEX_DENSITY:
            warnings.append(
                f"⚠️ WARN: LaTeX density {features['latex_density']:.2f} exceeds "
                f"threshold {MAX_LATEX_DENSITY}"
            )

        # Option count
        if features["option_count"] < MIN_OPTION_COUNT:
            warnings.append(
                f"⚠️ WARN: Only {features['option_count']} options — "
                f"recommended minimum is {MIN_OPTION_COUNT}"
            )

        # Readability
        if features["readability_score"] < MIN_READABILITY_SCORE:
            warnings.append(
                f"⚠️ WARN: Readability score {features['readability_score']} is low — "
                f"question may be hard to parse"
            )

        # Distractor similarity
        if features["distractor_similarity"] > 0.6:
            warnings.append(
                f"⚠️ WARN: High distractor similarity ({features['distractor_similarity']:.2f}) — "
                f"options may be too alike"
            )
        elif features["distractor_similarity"] < 0.05:
            warnings.append(
                f"ℹ️ INFO: Very low distractor similarity ({features['distractor_similarity']:.2f}) — "
                f"distractors may be too obvious"
            )

        # Negation
        if features.get("has_negation", 0):
            warnings.append(
                "ℹ️ INFO: Stem contains negation (NOT/EXCEPT) — "
                "can confuse students and reduce discrimination"
            )

        # Bloom's level
        if bloom_level in ["Remember", "Understand"]:
            warnings.append(
                f"ℹ️ INFO: Detected Bloom's level: {bloom_level} — "
                f"consider upgrading to Apply/Analyze for deeper assessment"
            )

        # Image references
        if features.get("has_image_reference", 0):
            warnings.append(
                "ℹ️ INFO: Question references a figure/image — "
                "ensure the visual is provided"
            )

        # Option length imbalance
        if features["answer_length_variance"] > 500:
            warnings.append(
                f"⚠️ WARN: High option length variance ({features['answer_length_variance']:.0f}) — "
                f"the longest option may be the obvious answer"
            )

        return warnings

    def get_bloom_level(self, question_text: str) -> str:
        """Get the estimated Bloom's Taxonomy level for a question."""
        return self._estimate_bloom_level(question_text)
