"""
XGBoost Prediction Wrapper for MCQ difficulty and discrimination.

Loads the pre-trained XGBoost models (JSON format) and provides
inference with SHAP-based feature importance.
"""
import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Optional
from config.settings import (
    XGB_CLF_MODEL_A, XGB_REG_MODEL_A,
    XGB_CLF_MODEL_B, XGB_REG_MODEL_B,
    DIFFICULTY_CLASSES, MODEL_A_NUM_FEATURES, MODEL_B_NUM_FEATURES
)
from ml.feature_extractor import FeatureExtractor


class QuestionPredictor:
    """Wraps XGBoost models for MCQ difficulty/discrimination prediction."""

    def __init__(self):
        """Load all XGBoost models from JSON files."""
        self.feature_extractor = FeatureExtractor()
        self._models_loaded = False
        
        # Model A: Text-only features (25)
        self.clf_a = None
        self.reg_a = None
        
        # Model B: All features (30)
        self.clf_b = None
        self.reg_b = None
        
        self._load_models()

    def _load_models(self):
        """Load XGBoost models from JSON files."""
        try:
            # Model A — Classifier (3-class: Easy/Medium/Hard)
            if XGB_CLF_MODEL_A.exists():
                self.clf_a = xgb.Booster()
                self.clf_a.load_model(str(XGB_CLF_MODEL_A))
            
            # Model A — Regressor (continuous difficulty index)
            if XGB_REG_MODEL_A.exists():
                self.reg_a = xgb.Booster()
                self.reg_a.load_model(str(XGB_REG_MODEL_A))
            
            # Model B — Classifier
            if XGB_CLF_MODEL_B.exists():
                self.clf_b = xgb.Booster()
                self.clf_b.load_model(str(XGB_CLF_MODEL_B))
            
            # Model B — Regressor
            if XGB_REG_MODEL_B.exists():
                self.reg_b = xgb.Booster()
                self.reg_b.load_model(str(XGB_REG_MODEL_B))
            
            self._models_loaded = any([self.clf_a, self.reg_a, self.clf_b, self.reg_b])
            
            if self._models_loaded:
                print(f"✅ Models loaded: "
                      f"CLF_A={'✓' if self.clf_a else '✗'} "
                      f"REG_A={'✓' if self.reg_a else '✗'} "
                      f"CLF_B={'✓' if self.clf_b else '✗'} "
                      f"REG_B={'✓' if self.reg_b else '✗'}")
            else:
                print("⚠️ No models found — will use synthetic fallback")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("   Falling back to synthetic predictions")
            self._models_loaded = False

    def predict(self, question_text: str, model: str = "A") -> dict:
        """Run full prediction pipeline on a question.
        
        Args:
            question_text: Raw MCQ text (stem + options).
            model: "A" (text-only) or "B" (all features).
            
        Returns:
            Dictionary with prediction results:
            {
                'difficulty': 'Hard',
                'difficulty_index': 0.82,
                'difficulty_probabilities': {'Easy': 0.05, 'Medium': 0.13, 'Hard': 0.82},
                'discrimination': 'Poor',
                'discrimination_index': 0.15,
                'feature_importance': {'latex_density': 0.34, ...},
                'features': {...},          # Raw feature values
                'confidence': 0.91,
                'model_used': 'A'
            }
        """
        # Extract features
        features = self.feature_extractor.extract_features(question_text, model=model)
        feature_names = list(features.keys())
        feature_values = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
        
        if self._models_loaded:
            return self._predict_with_models(feature_values, feature_names, features, model)
        else:
            return self._predict_synthetic(features, model)

    def _predict_with_models(self, feature_values: np.ndarray, 
                              feature_names: list, features: dict,
                              model: str) -> dict:
        """Predict using loaded XGBoost models."""
        dmatrix = xgb.DMatrix(feature_values, feature_names=feature_names)
        
        # Select models
        clf = self.clf_a if model == "A" else self.clf_b
        reg = self.reg_a if model == "A" else self.reg_b
        
        result = {
            "features": features,
            "model_used": model,
        }
        
        # Classification prediction (difficulty class)
        if clf:
            probs = clf.predict(dmatrix)[0]  # shape: (3,) for 3 classes
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            result["difficulty"] = DIFFICULTY_CLASSES[predicted_class]
            result["difficulty_probabilities"] = {
                DIFFICULTY_CLASSES[i]: round(float(probs[i]), 4)
                for i in range(len(DIFFICULTY_CLASSES))
            }
            result["confidence"] = round(confidence, 4)
            
            # Feature importance from the booster
            importance = clf.get_score(importance_type='weight')
            # Handle both named features and f0/f1... indexed features
            feature_importance = {}
            for key, val in importance.items():
                if key.startswith('f') and key[1:].isdigit():
                    # Indexed format: f0, f1, f2...
                    idx = int(key[1:])
                    if idx < len(feature_names):
                        feature_importance[feature_names[idx]] = round(val, 4)
                else:
                    # Named format: feature names stored in model
                    feature_importance[key] = round(val, 4)
            
            # Normalize to sum to 1
            total = sum(feature_importance.values()) or 1
            feature_importance = {
                k: round(v / total, 4) 
                for k, v in sorted(feature_importance.items(), key=lambda x: -x[1])
            }
            result["feature_importance"] = feature_importance
        else:
            result["difficulty"] = "Unknown"
            result["difficulty_probabilities"] = {}
            result["confidence"] = 0.0
            result["feature_importance"] = {}
        
        # Regression prediction (continuous difficulty index)
        if reg:
            reg_pred = reg.predict(dmatrix)[0]
            difficulty_index = float(np.clip(reg_pred, 0.0, 1.0))
            result["difficulty_index"] = round(difficulty_index, 4)
        else:
            # Estimate from classification probabilities
            result["difficulty_index"] = self._estimate_difficulty_index(
                result.get("difficulty", "Medium")
            )
        
        # Derive discrimination from difficulty index
        di = result["difficulty_index"]
        result["discrimination_index"] = self._estimate_discrimination(di, features)
        result["discrimination"] = self._classify_discrimination(
            result["discrimination_index"]
        )
        
        return result

    def _predict_synthetic(self, features: dict, model: str) -> dict:
        """Fallback: synthesize predictions from feature heuristics.
        
        Used when XGBoost models fail to load.
        """
        # Difficulty heuristic: combine multiple signals
        difficulty_score = 0.0
        
        # LaTeX density increases difficulty
        difficulty_score += features.get("latex_density", 0) * 2.0
        
        # Word count increases difficulty
        wc = features.get("word_count", 50)
        difficulty_score += min(wc / 200, 1.0) * 0.5
        
        # Low readability increases difficulty
        readability = features.get("readability_score", 60)
        difficulty_score += (1.0 - readability / 100) * 0.5
        
        # Negation increases difficulty
        difficulty_score += features.get("has_negation", 0) * 0.3
        
        # Higher Bloom's level increases difficulty
        bloom = features.get("bloom_level_encoded", 0)
        difficulty_score += bloom / 5.0 * 0.5
        
        # Formula count
        difficulty_score += min(features.get("formula_count", 0) / 5, 1.0) * 0.3
        
        # Normalize to 0-1
        difficulty_index = min(max(difficulty_score / 3.0, 0.05), 0.95)
        
        # Classify
        if difficulty_index < 0.35:
            difficulty = "Easy"
            probs = {"Easy": 0.7, "Medium": 0.2, "Hard": 0.1}
        elif difficulty_index < 0.65:
            difficulty = "Medium"
            probs = {"Easy": 0.15, "Medium": 0.7, "Hard": 0.15}
        else:
            difficulty = "Hard"
            probs = {"Easy": 0.05, "Medium": 0.2, "Hard": 0.75}
        
        discrimination_index = self._estimate_discrimination(difficulty_index, features)
        
        # Synthetic feature importance
        importance_raw = {
            "latex_density": features.get("latex_density", 0) * 2,
            "word_count": min(features.get("word_count", 0) / 200, 1),
            "text_complexity_score": features.get("text_complexity_score", 0) / 10,
            "vocab_richness": features.get("vocab_richness", 0.5),
            "avg_word_length": features.get("avg_word_length", 4) / 10,
            "answer_length_variance": min(features.get("answer_length_variance", 0) / 100, 1),
            "subject_difficulty_tier": features.get("subject_difficulty_tier", 0) / 2,
        }
        total = sum(importance_raw.values()) or 1
        feature_importance = {
            k: round(v / total, 4)
            for k, v in sorted(importance_raw.items(), key=lambda x: -x[1])
        }
        
        return {
            "difficulty": difficulty,
            "difficulty_index": round(difficulty_index, 4),
            "difficulty_probabilities": probs,
            "discrimination": self._classify_discrimination(discrimination_index),
            "discrimination_index": round(discrimination_index, 4),
            "feature_importance": feature_importance,
            "features": features,
            "confidence": max(probs.values()),
            "model_used": f"{model} (synthetic fallback)",
        }

    def _estimate_difficulty_index(self, difficulty_class: str) -> float:
        """Estimate a continuous difficulty index from a class label."""
        mapping = {"Easy": 0.25, "Medium": 0.50, "Hard": 0.80}
        return mapping.get(difficulty_class, 0.50)

    def _estimate_discrimination(self, difficulty_index: float, features: dict) -> float:
        """Estimate discrimination index.
        
        Best discrimination at medium difficulty (di ≈ 0.5),
        poor at extremes (too easy or too hard).
        """
        # Parabolic model: max discrimination at 0.5
        base = 1.0 - 4.0 * (difficulty_index - 0.5) ** 2
        
        # Adjust for distractor similarity
        similarity = features.get("distractor_similarity", 0.3)
        # High similarity = better distractors = better discrimination
        base *= (0.5 + similarity)
        
        # Negation tends to reduce discrimination
        if features.get("has_negation", 0):
            base *= 0.8
        
        return round(min(max(base, 0.05), 0.95), 4)

    def _classify_discrimination(self, discrimination_index: float) -> str:
        """Classify discrimination index into categories."""
        if discrimination_index >= 0.40:
            return "Excellent"
        elif discrimination_index >= 0.30:
            return "Good"
        elif discrimination_index >= 0.20:
            return "Fair"
        else:
            return "Poor"

    def validate_question(self, question_text: str, 
                          target_difficulty: str,
                          model: str = "A") -> dict:
        """Validate whether a generated question matches the target difficulty.
        
        Used by the Refiner node to verify LLM-generated questions.
        
        Returns:
            {
                'matches': True/False,
                'predicted': 'Hard',
                'target': 'Medium',
                'prediction': {...full prediction dict...}
            }
        """
        prediction = self.predict(question_text, model=model)
        matches = prediction["difficulty"] == target_difficulty
        
        return {
            "matches": matches,
            "predicted": prediction["difficulty"],
            "target": target_difficulty,
            "prediction": prediction,
        }
