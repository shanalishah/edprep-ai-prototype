import re
import os
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

class EnhancedEssayScorer:
    """
    Enhanced AI-powered essay scorer with proper gibberish detection and ML models
    """
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self._load_models()
        
        # Gibberish detection patterns
        self.gibberish_patterns = [
            r'[a-z]{1,2}[A-Z][a-z]{1,2}[A-Z]',  # Mixed case patterns
            r'(.)\1{4,}',  # Repeated characters (5+ times)
            r'[^a-zA-Z\s.,!?;:\'"()-]{3,}',  # Non-alphabetic sequences
        ]
        
        # Common English words for quality assessment
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
            'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up',
            'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
            'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
            'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think',
            'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
            'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are'
        }
    
    def _load_models(self):
        """Load ML models if available"""
        try:
            models_path = os.path.join(os.path.dirname(__file__), '../../models')
            
            # Try to load production models
            if os.path.exists(os.path.join(models_path, 'Production_Random_Forest_model.pkl')):
                self.models['random_forest'] = joblib.load(os.path.join(models_path, 'Production_Random_Forest_model.pkl'))
                print("✅ Loaded Random Forest model")
            
            if os.path.exists(os.path.join(models_path, 'production_tfidf_vectorizer.pkl')):
                self.vectorizers['tfidf'] = joblib.load(os.path.join(models_path, 'production_tfidf_vectorizer.pkl'))
                print("✅ Loaded TF-IDF vectorizer")
                
            if os.path.exists(os.path.join(models_path, 'production_scaler.pkl')):
                self.scalers['scaler'] = joblib.load(os.path.join(models_path, 'production_scaler.pkl'))
                print("✅ Loaded scaler")
                
        except Exception as e:
            print(f"⚠️ Could not load ML models: {e}")
            print("🚀 Using enhanced rule-based scoring")
    
    def is_gibberish_or_low_quality(self, essay: str) -> bool:
        """Detect if essay is gibberish or extremely low quality"""
        if not essay or len(essay.strip()) < 10:
            return True
            
        # Check for repeated patterns
        for pattern in self.gibberish_patterns:
            if re.search(pattern, essay):
                return True
        
        # Check word quality
        words = re.findall(r'\b\w+\b', essay.lower())
        if len(words) < 5:
            return True
            
        # Check for meaningful words ratio
        meaningful_words = sum(1 for word in words if word in self.common_words or len(word) > 3)
        if len(words) > 0 and meaningful_words / len(words) < 0.3:
            return True
            
        # Check for excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, it's likely gibberish
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition / len(words) > 0.3:
            return True
            
        return False
    
    def score_essay(self, prompt: str, essay: str, task_type: str = "Task 2") -> Dict[str, float]:
        """Score an essay with enhanced detection"""
        
        # Check for gibberish first
        if self.is_gibberish_or_low_quality(essay):
            return {
                "task_achievement": 1.0,
                "coherence_cohesion": 1.0,
                "lexical_resource": 1.0,
                "grammatical_range": 1.0,
                "overall_band_score": 1.0
            }
        
        # Try ML scoring first, fallback to rule-based
        if self.models and 'random_forest' in self.models:
            try:
                return self._ml_score_essay(prompt, essay, task_type)
            except Exception as e:
                print(f"ML scoring failed: {e}, using rule-based")
        
        return self._rule_based_score_essay(prompt, essay, task_type)
    
    def _ml_score_essay(self, prompt: str, essay: str, task_type: str) -> Dict[str, float]:
        """Use ML models for scoring"""
        # Extract features
        features = self._extract_features(essay, prompt, task_type)
        
        # Vectorize text
        if 'tfidf' in self.vectorizers:
            essay_vector = self.vectorizers['tfidf'].transform([essay])
            features = np.hstack([features, essay_vector.toarray()])
        
        # Scale features
        if 'scaler' in self.scalers:
            features = self.scalers['scaler'].transform(features.reshape(1, -1))
        
        # Predict
        model = self.models['random_forest']
        prediction = model.predict(features)[0]
        
        # Convert to individual scores (simplified)
        base_score = max(1.0, min(9.0, prediction))
        
        return {
            "task_achievement": base_score,
            "coherence_cohesion": base_score,
            "lexical_resource": base_score,
            "grammatical_range": base_score,
            "overall_band_score": base_score
        }
    
    def _rule_based_score_essay(self, prompt: str, essay: str, task_type: str) -> Dict[str, float]:
        """Enhanced rule-based scoring"""
        task_achievement = self._score_task_achievement(prompt, essay, task_type)
        coherence_cohesion = self._score_coherence_cohesion(essay)
        lexical_resource = self._score_lexical_resource(essay)
        grammatical_range = self._score_grammatical_range(essay)
        
        overall_band_score = (task_achievement + coherence_cohesion + lexical_resource + grammatical_range) / 4
        
        return {
            "task_achievement": task_achievement,
            "coherence_cohesion": coherence_cohesion,
            "lexical_resource": lexical_resource,
            "grammatical_range": grammatical_range,
            "overall_band_score": overall_band_score
        }
    
    def _extract_features(self, essay: str, prompt: str, task_type: str) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        
        # Word count
        word_count = len(essay.split())
        features.append(word_count)
        
        # Sentence count
        sentence_count = len(re.split(r'[.!?]+', essay))
        features.append(sentence_count)
        
        # Average sentence length
        features.append(word_count / max(1, sentence_count))
        
        # Character count
        features.append(len(essay))
        
        # Paragraph count
        paragraph_count = len([p for p in essay.split('\n') if p.strip()])
        features.append(paragraph_count)
        
        # Linking words count
        linking_words = ['however', 'moreover', 'furthermore', 'therefore', 'consequently', 'nevertheless']
        linking_count = sum(1 for word in linking_words if word in essay.lower())
        features.append(linking_count)
        
        # Task type (binary)
        features.append(1 if task_type == "Task 2" else 0)
        
        return np.array(features)
    
    def _score_task_achievement(self, prompt: str, essay: str, task_type: str) -> float:
        """Enhanced task achievement scoring"""
        score = 3.0  # Start lower
        
        word_count = len(essay.split())
        
        # Strict word count requirements
        if task_type == "Task 2":
            if word_count < 250:
                score = max(1.0, score - (250 - word_count) / 50)  # Penalty for short essays
            elif word_count >= 250:
                score += 1.0
            if word_count > 400:  # Too long
                score -= 0.5
        else:  # Task 1
            if word_count < 150:
                score = max(1.0, score - (150 - word_count) / 30)
            elif word_count >= 150:
                score += 1.0
            if word_count > 250:
                score -= 0.5
        
        # Check for clear position (Task 2)
        if task_type == "Task 2":
            opinion_indicators = ["i believe", "i think", "in my opinion", "i agree", "i disagree", "i support", "i oppose"]
            if any(indicator in essay.lower() for indicator in opinion_indicators):
                score += 1.0
            else:
                score -= 0.5
        
        # Check for examples and evidence
        example_indicators = ["for example", "for instance", "such as", "like", "specifically", "particularly"]
        if any(indicator in essay.lower() for indicator in example_indicators):
            score += 0.5
        
        # Check for conclusion
        conclusion_indicators = ["in conclusion", "to conclude", "to sum up", "overall", "in summary"]
        if any(indicator in essay.lower() for indicator in conclusion_indicators):
            score += 0.5
        
        return min(9.0, max(1.0, score))
    
    def _score_coherence_cohesion(self, essay: str) -> float:
        """Enhanced coherence and cohesion scoring"""
        score = 3.0
        
        # Check for paragraph structure
        paragraphs = [p.strip() for p in essay.split('\n') if p.strip()]
        if len(paragraphs) >= 3:  # Introduction, body, conclusion
            score += 1.0
        elif len(paragraphs) < 2:
            score -= 1.0
        
        # Check for linking words
        linking_words = ['however', 'moreover', 'furthermore', 'therefore', 'consequently', 'nevertheless', 
                        'additionally', 'similarly', 'likewise', 'in contrast', 'on the other hand']
        linking_count = sum(1 for word in linking_words if word in essay.lower())
        score += min(1.0, linking_count * 0.2)
        
        # Check for topic sentences (simplified)
        sentences = re.split(r'[.!?]+', essay)
        topic_sentence_indicators = ['first', 'second', 'third', 'initially', 'furthermore', 'moreover', 'however']
        topic_sentences = sum(1 for sentence in sentences if any(indicator in sentence.lower() for indicator in topic_sentence_indicators))
        score += min(0.5, topic_sentences * 0.1)
        
        return min(9.0, max(1.0, score))
    
    def _score_lexical_resource(self, essay: str) -> float:
        """Enhanced lexical resource scoring"""
        score = 3.0
        
        words = re.findall(r'\b\w+\b', essay.lower())
        unique_words = set(words)
        
        # Lexical diversity
        if len(words) > 0:
            diversity = len(unique_words) / len(words)
            if diversity > 0.7:
                score += 1.5
            elif diversity > 0.5:
                score += 0.5
            else:
                score -= 0.5
        
        # Check for advanced vocabulary
        advanced_words = ['consequently', 'furthermore', 'nevertheless', 'substantially', 'significantly', 
                         'considerably', 'remarkably', 'particularly', 'specifically', 'fundamentally']
        advanced_count = sum(1 for word in advanced_words if word in essay.lower())
        score += min(1.0, advanced_count * 0.2)
        
        # Check for word repetition (penalty)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition / len(words) > 0.1:  # More than 10% repetition
            score -= 0.5
        
        return min(9.0, max(1.0, score))
    
    def _score_grammatical_range(self, essay: str) -> float:
        """Enhanced grammatical range and accuracy scoring"""
        score = 3.0
        
        sentences = re.split(r'[.!?]+', essay)
        if len(sentences) < 3:
            score -= 1.0
        
        # Check for sentence variety
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            if 10 <= avg_length <= 25:  # Good sentence length
                score += 0.5
            elif avg_length < 5:  # Too short
                score -= 0.5
        
        # Check for complex sentences
        complex_indicators = ['because', 'although', 'while', 'whereas', 'since', 'if', 'unless', 'until']
        complex_count = sum(1 for indicator in complex_indicators if indicator in essay.lower())
        score += min(1.0, complex_count * 0.2)
        
        # Check for passive voice
        passive_indicators = ['is', 'are', 'was', 'were', 'been', 'being']
        passive_count = sum(1 for indicator in passive_indicators if indicator in essay.lower())
        if passive_count > 0:
            score += 0.3  # Some passive voice is good
        
        # Basic grammar check (simplified)
        # Check for proper capitalization
        sentences_proper = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if len(sentences) > 0 and sentences_proper / len(sentences) > 0.8:
            score += 0.5
        
        return min(9.0, max(1.0, score))
