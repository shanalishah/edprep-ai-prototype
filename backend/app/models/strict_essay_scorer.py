"""
Strict Essay Scorer: Advanced ML Models for Real IELTS-like Assessment
Uses the locally trained models for strict and accurate scoring
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional

class StrictEssayScorer:
    """
    Strict scoring system using locally trained advanced models
    """
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.tfidf_vectorizer = None
        self.feature_columns = None
        self.models = {}
        self.best_models = {}
        self.is_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """
        Load all trained models
        """
        try:
            # Load TF-IDF vectorizer
            vectorizer_path = self.models_dir / "strict_tfidf_vectorizer.pkl"
            if vectorizer_path.exists():
                self.tfidf_vectorizer = joblib.load(vectorizer_path)
                print(f"✅ Strict TF-IDF Vectorizer loaded")
            else:
                print(f"⚠️ Strict TF-IDF Vectorizer not found")
                return
            
            # Load feature columns
            columns_path = self.models_dir / "strict_feature_columns.pkl"
            if columns_path.exists():
                self.feature_columns = joblib.load(columns_path)
                print(f"✅ Strict feature columns loaded")
            else:
                print(f"⚠️ Strict feature columns not found")
                return
            
            # Load best models info
            best_models_path = self.models_dir / "strict_best_models.pkl"
            if best_models_path.exists():
                self.best_models = joblib.load(best_models_path)
                print(f"✅ Best models info loaded")
            else:
                print(f"⚠️ Best models info not found")
                return
            
            # Load models for each criterion
            criteria = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
            
            for criterion in criteria:
                criterion_dir = self.models_dir / f"strict_{criterion}"
                if criterion_dir.exists():
                    self.models[criterion] = {}
                    
                    # Load the best model for this criterion
                    best_model_name = self.best_models.get(criterion)
                    if best_model_name:
                        model_path = criterion_dir / f"{best_model_name}_model.pkl"
                        scaler_path = criterion_dir / f"{best_model_name}_scaler.pkl"
                        
                        if model_path.exists():
                            self.models[criterion]['model'] = joblib.load(model_path)
                            
                            if scaler_path.exists():
                                self.models[criterion]['scaler'] = joblib.load(scaler_path)
                            else:
                                self.models[criterion]['scaler'] = None
                            
                            print(f"✅ Loaded {criterion}/{best_model_name}")
                        else:
                            print(f"⚠️ Model not found: {criterion}/{best_model_name}")
                    else:
                        print(f"⚠️ Best model not specified for {criterion}")
                else:
                    print(f"⚠️ Criterion directory not found: {criterion}")
            
            self.is_loaded = True
            print("✅ Strict models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading strict models: {e}")
            self.is_loaded = False
    
    def _extract_features(self, essay: str, prompt: str) -> pd.DataFrame:
        """
        Extract features for scoring
        """
        # Basic features
        word_count = len(essay.split())
        sentence_count = len(re.split(r'[.!?]', essay))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        paragraph_count = essay.count('\n\n') + 1
        
        # Advanced linguistic features
        complex_words = len([w for w in essay.split() if len(w) > 6])
        complexity_ratio = complex_words / word_count if word_count > 0 else 0
        
        # Cohesion features
        cohesive_markers = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 
                          'in addition', 'on the other hand', 'as a result', 'for example', 'for instance']
        cohesion_score = sum(essay.lower().count(marker) for marker in cohesive_markers)
        
        # Task-specific features
        prompt_keywords = set(re.findall(r'\b\w+\b', prompt.lower()))
        essay_keywords = set(re.findall(r'\b\w+\b', essay.lower()))
        keyword_overlap = len(prompt_keywords.intersection(essay_keywords))
        
        # Readability features
        avg_word_length = np.mean([len(word) for word in essay.split() if word.isalpha()]) if word_count > 0 else 0
        
        # Grammar and structure features
        question_marks = essay.count('?')
        exclamation_marks = essay.count('!')
        commas = essay.count(',')
        semicolons = essay.count(';')
        colons = essay.count(':')
        
        # Essay structure analysis
        intro_indicators = ['introduction', 'firstly', 'to begin', 'initially']
        conclusion_indicators = ['conclusion', 'finally', 'in conclusion', 'to conclude', 'in summary']
        
        has_intro = any(indicator in essay.lower() for indicator in intro_indicators)
        has_conclusion = any(indicator in essay.lower() for indicator in conclusion_indicators)
        
        # Create basic features DataFrame
        basic_features = pd.DataFrame([{
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'paragraph_count': paragraph_count,
            'complexity_ratio': complexity_ratio,
            'cohesion_score': cohesion_score,
            'keyword_overlap': keyword_overlap,
            'avg_word_length': avg_word_length,
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks,
            'commas': commas,
            'semicolons': semicolons,
            'colons': colons,
            'has_intro': int(has_intro),
            'has_conclusion': int(has_conclusion)
        }])
        
        # Create TF-IDF features
        if self.tfidf_vectorizer:
            cleaned_essay = re.sub(r'[^\w\s]', ' ', essay.lower())
            cleaned_essay = re.sub(r'\s+', ' ', cleaned_essay).strip()
            
            tfidf_features = self.tfidf_vectorizer.transform([cleaned_essay]).toarray()
            tfidf_df = pd.DataFrame(
                tfidf_features,
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            
            # Combine features
            combined_features = pd.concat([basic_features, tfidf_df], axis=1)
        else:
            combined_features = basic_features
        
        return combined_features
    
    def score_essay_strict(self, essay: str, prompt: str, task_type: str = "Task 2") -> Dict[str, float]:
        """
        Score essay using strict trained models
        """
        if not self.is_loaded:
            print("❌ Strict models not loaded")
            return {}
        
        try:
            # Extract features
            features_df = self._extract_features(essay, prompt)
            
            # Ensure we have the right number of features
            if len(features_df.columns) != len(self.feature_columns) + self.tfidf_vectorizer.max_features:
                print(f"⚠️ Feature mismatch: got {len(features_df.columns)}, expected {len(self.feature_columns) + self.tfidf_vectorizer.max_features}")
                # Pad or truncate features to match
                expected_cols = self.feature_columns + [f'tfidf_{i}' for i in range(self.tfidf_vectorizer.max_features)]
                for col in expected_cols:
                    if col not in features_df.columns:
                        features_df[col] = 0
                features_df = features_df[expected_cols]
            
            scores = {}
            
            # Score each criterion
            for criterion in ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']:
                if criterion in self.models and 'model' in self.models[criterion]:
                    model = self.models[criterion]['model']
                    scaler = self.models[criterion].get('scaler')
                    
                    # Prepare features
                    if scaler:
                        features_scaled = scaler.transform(features_df)
                        prediction = model.predict(features_scaled)[0]
                    else:
                        prediction = model.predict(features_df)[0]
                    
                    # Apply strict scoring adjustments
                    if criterion == 'overall_band_score':
                        # Apply strict penalties for low-quality essays
                        word_count = len(essay.split())
                        if word_count < 250 and task_type == "Task 2":
                            prediction -= 0.5  # Penalty for short essays
                        elif word_count < 200:
                            prediction -= 1.0  # Heavy penalty for very short essays
                        
                        # Apply strict penalties for poor structure
                        sentence_count = len(re.split(r'[.!?]', essay))
                        if sentence_count < 5:
                            prediction -= 0.5  # Penalty for poor structure
                    
                    # Round to nearest 0.5 band score
                    scores[criterion] = round(max(1.0, min(9.0, prediction)) * 2) / 2
                else:
                    print(f"⚠️ Model not available for {criterion}")
                    scores[criterion] = 6.0  # Default score
            
            return scores
            
        except Exception as e:
            print(f"❌ Error during strict scoring: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        """
        if not self.is_loaded:
            return {"error": "Models not loaded"}
        
        return {
            "is_loaded": self.is_loaded,
            "total_features": len(self.feature_columns) + self.tfidf_vectorizer.max_features if self.tfidf_vectorizer else len(self.feature_columns),
            "basic_features": len(self.feature_columns),
            "tfidf_features": self.tfidf_vectorizer.max_features if self.tfidf_vectorizer else 0,
            "available_criteria": list(self.models.keys()),
            "best_models": self.best_models
        }
