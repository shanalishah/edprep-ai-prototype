import re
from typing import Dict, List, Tuple
import os

class EssayScorer:
    """
    AI-powered essay scorer for IELTS writing assessment
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._load_models()
    
    def _load_models(self):
        """
        Load pre-trained models for each scoring criterion
        For now, we'll use a simple rule-based approach
        In production, these would be fine-tuned models
        """
        print("Loading essay scoring models...")
        # TODO: Load actual fine-tuned models
        # For now, we'll implement rule-based scoring
        pass
    
    def score_essay(self, prompt: str, essay: str, task_type: str = "Task 2") -> Dict[str, float]:
        """
        Score an essay based on IELTS criteria
        """
        try:
            # Calculate scores for each criterion
            task_achievement = self._score_task_achievement(prompt, essay, task_type)
            coherence_cohesion = self._score_coherence_cohesion(essay)
            lexical_resource = self._score_lexical_resource(essay)
            grammatical_range = self._score_grammatical_range(essay)
            
            # Calculate overall band score
            overall_band_score = self._calculate_overall_score([
                task_achievement, coherence_cohesion, lexical_resource, grammatical_range
            ])
            
            return {
                "task_achievement": task_achievement,
                "coherence_cohesion": coherence_cohesion,
                "lexical_resource": lexical_resource,
                "grammatical_range": grammatical_range,
                "overall_band_score": overall_band_score
            }
            
        except Exception as e:
            print(f"Error scoring essay: {e}")
            # Return default scores if error occurs
            return {
                "task_achievement": 5.0,
                "coherence_cohesion": 5.0,
                "lexical_resource": 5.0,
                "grammatical_range": 5.0,
                "overall_band_score": 5.0
            }
    
    def _score_task_achievement(self, prompt: str, essay: str, task_type: str) -> float:
        """
        Score Task Achievement based on how well the essay addresses the prompt
        """
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
        """
        Score Coherence and Cohesion based on organization and linking
        """
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
        """
        Score Lexical Resource based on vocabulary range and accuracy
        """
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
        """
        Score Grammatical Range and Accuracy
        """
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
        """
        Calculate overall band score (average rounded to nearest 0.5)
        """
        average = sum(scores) / len(scores)
        # Round to nearest 0.5
        rounded = round(average * 2) / 2
        return min(9.0, max(1.0, rounded))

