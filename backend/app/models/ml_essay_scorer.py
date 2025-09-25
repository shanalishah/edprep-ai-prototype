"""
ML-based Essay Scorer for IELTS Writing Assessment
This module integrates the trained Random Forest model into the API
"""

import pickle
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Try to download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MLEssayScorer:
    """
    ML-powered essay scorer using trained Random Forest model
    """
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default path relative to this file
            models_dir = Path(__file__).parent.parent.parent.parent / "models"
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load the trained model and scaler"""
        try:
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load Random Forest model
            model_path = self.models_dir / "Random Forest_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            self.is_loaded = self.model is not None and self.scaler is not None
            
            if self.is_loaded:
                print("✅ ML models loaded successfully")
            else:
                print("⚠️  ML models not found, falling back to rule-based scoring")
                
        except Exception as e:
            print(f"❌ Error loading ML models: {e}")
            self.is_loaded = False
    
    def extract_text_features(self, essay: str) -> np.ndarray:
        """Extract the same features used during training"""
        if not essay or pd.isna(essay):
            return self._get_empty_features()
        
        essay = str(essay)
        
        # Basic features
        word_count = len(essay.split())
        char_count = len(essay)
        sentence_count = len(sent_tokenize(essay))
        paragraph_count = len([p for p in essay.split('\n\n') if p.strip()])
        
        # Word-level features
        words = word_tokenize(essay.lower())
        unique_words = len(set(words))
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Sentence-level features
        sentences = sent_tokenize(essay)
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        # Vocabulary features
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        # Grammar and style features
        complex_sentences = len(re.findall(r'\b(because|although|while|whereas|if|when|since|as)\b', essay.lower()))
        linking_words = len(re.findall(r'\b(however|therefore|moreover|furthermore|additionally|on the other hand|in contrast|similarly|likewise|firstly|secondly|finally|in addition|as a result)\b', essay.lower()))
        
        # Academic vocabulary
        academic_words = len(re.findall(r'\b(significant|substantial|considerable|essential|crucial|fundamental|comprehensive|extensive|effective|efficient|consequently|subsequently|furthermore|moreover|nevertheless)\b', essay.lower()))
        
        # Punctuation features
        comma_count = essay.count(',')
        period_count = essay.count('.')
        question_count = essay.count('?')
        exclamation_count = essay.count('!')
        
        # Capitalization features
        capital_ratio = sum(1 for c in essay if c.isupper()) / len(essay) if essay else 0
        
        # Repetition features
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only count longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        max_word_freq = max(word_freq.values()) if word_freq else 0
        repetition_ratio = max_word_freq / word_count if word_count > 0 else 0
        
        # Additional features from existing dataset
        essay_length = len(essay)
        
        features = np.array([
            word_count, char_count, sentence_count, paragraph_count,
            unique_words, avg_word_length, avg_sentence_length,
            vocabulary_richness, complex_sentences, linking_words,
            academic_words, comma_count, period_count, question_count,
            exclamation_count, capital_ratio, repetition_ratio,
            essay_length, word_count, essay_length, avg_word_length,
            sentence_count, paragraph_count, complex_sentences, linking_words
        ])
        
        return features.reshape(1, -1)
    
    def _get_empty_features(self) -> np.ndarray:
        """Return empty features for null essays"""
        return np.zeros((1, 25))
    
    def score_essay_ml(self, essay: str, prompt: str = "", task_type: str = "Task 2") -> Dict[str, float]:
        """
        Score essay using ML model
        """
        if not self.is_loaded:
            # Fallback to rule-based scoring
            return self._fallback_scoring(essay, prompt, task_type)
        
        try:
            # Extract features
            features = self.extract_text_features(essay)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            predictions = self.model.predict(features_scaled)[0]
            
            # Map predictions to score names
            scores = {
                'task_achievement': float(predictions[0]),
                'coherence_cohesion': float(predictions[1]),
                'lexical_resource': float(predictions[2]),
                'grammatical_range': float(predictions[3]),
                'overall_band_score': float(predictions[4])
            }
            
            # Ensure scores are within valid range
            for key in scores:
                scores[key] = max(1.0, min(9.0, scores[key]))
            
            # Calculate overall band score as average if not provided
            if scores['overall_band_score'] is None or scores['overall_band_score'] == 0:
                overall = (scores['task_achievement'] + scores['coherence_cohesion'] + 
                          scores['lexical_resource'] + scores['grammatical_range']) / 4
                scores['overall_band_score'] = round(overall * 2) / 2  # Round to nearest 0.5
            
            return scores
            
        except Exception as e:
            print(f"Error in ML scoring: {e}")
            return self._fallback_scoring(essay, prompt, task_type)
    
    def _fallback_scoring(self, essay: str, prompt: str, task_type: str) -> Dict[str, float]:
        """
        Fallback rule-based scoring when ML model is not available
        """
        # Simple rule-based scoring (from original essay_scorer.py)
        task_achievement = self._score_task_achievement(prompt, essay, task_type)
        coherence_cohesion = self._score_coherence_cohesion(essay)
        lexical_resource = self._score_lexical_resource(essay)
        grammatical_range = self._score_grammatical_range(essay)
        
        # Calculate overall band score
        overall_band_score = self._calculate_overall_score([
            task_achievement, coherence_cohesion, lexical_resource, grammatical_range
        ])
        
        return {
            'task_achievement': task_achievement,
            'coherence_cohesion': coherence_cohesion,
            'lexical_resource': lexical_resource,
            'grammatical_range': grammatical_range,
            'overall_band_score': overall_band_score
        }
    
    def _score_task_achievement(self, prompt: str, essay: str, task_type: str) -> float:
        """Score Task Achievement based on how well the essay addresses the prompt"""
        score = 5.0  # Base score
        
        # Check word count
        word_count = len(essay.split())
        if task_type == "Task 2":
            if word_count >= 250:
                score += 1.0
            elif word_count < 150:
                score -= 1.5
        else:  # Task 1
            if word_count >= 150:
                score += 1.0
            elif word_count < 100:
                score -= 1.5
        
        # Check for clear position/opinion (for Task 2)
        if task_type == "Task 2":
            opinion_indicators = ["I believe", "I think", "In my opinion", "I agree", "I disagree"]
            if any(indicator in essay.lower() for indicator in opinion_indicators):
                score += 0.5
        
        # Check for examples and evidence
        example_indicators = ["for example", "for instance", "such as", "like"]
        if any(indicator in essay.lower() for indicator in example_indicators):
            score += 0.5
        
        # Check for conclusion
        conclusion_indicators = ["in conclusion", "to conclude", "to sum up", "overall"]
        if any(indicator in essay.lower() for indicator in conclusion_indicators):
            score += 0.5
        
        return min(9.0, max(1.0, score))
    
    def _score_coherence_cohesion(self, essay: str) -> float:
        """Score Coherence and Cohesion based on organization and linking"""
        score = 5.0  # Base score
        
        # Check for paragraph structure
        paragraphs = essay.split('\n\n')
        if len(paragraphs) >= 3:  # Introduction, body, conclusion
            score += 1.0
        
        # Check for linking words
        linking_words = [
            "however", "therefore", "moreover", "furthermore", "additionally",
            "on the other hand", "in contrast", "similarly", "likewise",
            "firstly", "secondly", "finally", "in addition", "as a result"
        ]
        
        linking_count = sum(1 for word in linking_words if word in essay.lower())
        if linking_count >= 3:
            score += 1.0
        elif linking_count >= 1:
            score += 0.5
        
        # Check for topic sentences (simple heuristic)
        sentences = essay.split('.')
        if len(sentences) >= 5:
            score += 0.5
        
        return min(9.0, max(1.0, score))
    
    def _score_lexical_resource(self, essay: str) -> float:
        """Score Lexical Resource based on vocabulary range and accuracy"""
        score = 5.0  # Base score
        
        # Calculate vocabulary diversity (unique words / total words)
        words = essay.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            diversity = len(unique_words) / len(words)
            if diversity > 0.7:
                score += 1.5
            elif diversity > 0.6:
                score += 1.0
            elif diversity > 0.5:
                score += 0.5
        
        # Check for academic vocabulary
        academic_words = [
            "significant", "substantial", "considerable", "essential", "crucial",
            "fundamental", "comprehensive", "extensive", "effective", "efficient",
            "consequently", "subsequently", "furthermore", "moreover", "nevertheless"
        ]
        
        academic_count = sum(1 for word in academic_words if word in essay.lower())
        if academic_count >= 3:
            score += 1.0
        elif academic_count >= 1:
            score += 0.5
        
        # Check for word repetition (penalty)
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only count longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        max_repetition = max(word_freq.values()) if word_freq else 0
        if max_repetition > 5:
            score -= 0.5
        
        return min(9.0, max(1.0, score))
    
    def _score_grammatical_range(self, essay: str) -> float:
        """Score Grammatical Range and Accuracy"""
        score = 5.0  # Base score
        
        # Check for sentence variety
        sentences = [s.strip() for s in essay.split('.') if s.strip()]
        if len(sentences) >= 5:
            # Check for complex sentences
            complex_indicators = ["because", "although", "while", "whereas", "if", "when"]
            complex_count = sum(1 for sentence in sentences 
                             if any(indicator in sentence.lower() for indicator in complex_indicators))
            
            if complex_count >= 2:
                score += 1.0
            elif complex_count >= 1:
                score += 0.5
        
        # Check for passive voice
        passive_indicators = ["is", "are", "was", "were", "been", "being"]
        passive_count = sum(1 for sentence in sentences 
                          if any(indicator in sentence.lower() for indicator in passive_indicators))
        if passive_count >= 1:
            score += 0.5
        
        # Simple grammar error detection
        common_errors = [
            r'\bi\b',  # lowercase 'i' should be 'I'
            r'\bthe the\b',  # double 'the'
            r'\ba a\b',  # double 'a'
        ]
        
        error_count = 0
        for pattern in common_errors:
            error_count += len(re.findall(pattern, essay))
        
        if error_count == 0:
            score += 0.5
        elif error_count <= 2:
            score += 0.2
        else:
            score -= 0.5
        
        return min(9.0, max(1.0, score))
    
    def _calculate_overall_score(self, scores: List[float]) -> float:
        """Calculate overall band score (average rounded to nearest 0.5)"""
        average = sum(scores) / len(scores)
        # Round to nearest 0.5
        rounded = round(average * 2) / 2
        return min(9.0, max(1.0, rounded))
