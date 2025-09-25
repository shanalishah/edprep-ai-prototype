"""
Realistic IELTS Essay Scorer
Implements strict, authentic IELTS scoring based on official criteria
"""

import re
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import Counter
import math

# Simple tokenization without NLTK dependencies
def simple_word_tokenize(text: str) -> List[str]:
    """Simple word tokenization"""
    return re.findall(r'\b\w+\b', text.lower())

def simple_sent_tokenize(text: str) -> List[str]:
    """Simple sentence tokenization"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

# Common English stopwords
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once'
}

class RealisticIELTSScorer:
    """
    Strict, realistic IELTS essay scorer based on official criteria
    Implements authentic scoring that matches real IELTS standards
    """
    
    def __init__(self):
        self.stop_words = STOP_WORDS
        
        # IELTS Band Descriptors (simplified but realistic)
        self.band_descriptors = {
            'task_achievement': {
                9: {'words': 'fully addresses', 'development': 'fully extended', 'position': 'clear'},
                8: {'words': 'sufficiently addresses', 'development': 'well extended', 'position': 'clear'},
                7: {'words': 'addresses', 'development': 'extended', 'position': 'clear'},
                6: {'words': 'addresses', 'development': 'relevant', 'position': 'clear'},
                5: {'words': 'partially addresses', 'development': 'limited', 'position': 'unclear'},
                4: {'words': 'minimally addresses', 'development': 'minimal', 'position': 'unclear'},
                3: {'words': 'barely addresses', 'development': 'very limited', 'position': 'unclear'},
                2: {'words': 'does not address', 'development': 'irrelevant', 'position': 'unclear'},
                1: {'words': 'completely off-topic', 'development': 'irrelevant', 'position': 'unclear'}
            },
            'coherence_cohesion': {
                9: {'structure': 'clear progression', 'cohesion': 'skillful', 'paragraphing': 'effective'},
                8: {'structure': 'clear progression', 'cohesion': 'good', 'paragraphing': 'effective'},
                7: {'structure': 'clear progression', 'cohesion': 'adequate', 'paragraphing': 'logical'},
                6: {'structure': 'some progression', 'cohesion': 'adequate', 'paragraphing': 'adequate'},
                5: {'structure': 'limited progression', 'cohesion': 'inadequate', 'paragraphing': 'inadequate'},
                4: {'structure': 'minimal progression', 'cohesion': 'minimal', 'paragraphing': 'minimal'},
                3: {'structure': 'no clear progression', 'cohesion': 'minimal', 'paragraphing': 'minimal'},
                2: {'structure': 'no progression', 'cohesion': 'none', 'paragraphing': 'none'},
                1: {'structure': 'no progression', 'cohesion': 'none', 'paragraphing': 'none'}
            },
            'lexical_resource': {
                9: {'range': 'wide', 'accuracy': 'natural', 'collocation': 'skillful'},
                8: {'range': 'wide', 'accuracy': 'occasional errors', 'collocation': 'good'},
                7: {'range': 'sufficient', 'accuracy': 'occasional errors', 'collocation': 'adequate'},
                6: {'range': 'adequate', 'accuracy': 'some errors', 'collocation': 'adequate'},
                5: {'range': 'limited', 'accuracy': 'frequent errors', 'collocation': 'limited'},
                4: {'range': 'limited', 'accuracy': 'frequent errors', 'collocation': 'limited'},
                3: {'range': 'very limited', 'accuracy': 'many errors', 'collocation': 'very limited'},
                2: {'range': 'very limited', 'accuracy': 'many errors', 'collocation': 'very limited'},
                1: {'range': 'extremely limited', 'accuracy': 'many errors', 'collocation': 'extremely limited'}
            },
            'grammatical_range': {
                9: {'range': 'wide', 'accuracy': 'natural', 'complexity': 'skillful'},
                8: {'range': 'wide', 'accuracy': 'occasional errors', 'complexity': 'good'},
                7: {'range': 'sufficient', 'accuracy': 'occasional errors', 'complexity': 'adequate'},
                6: {'range': 'adequate', 'accuracy': 'some errors', 'complexity': 'adequate'},
                5: {'range': 'limited', 'accuracy': 'frequent errors', 'complexity': 'limited'},
                4: {'range': 'limited', 'accuracy': 'frequent errors', 'complexity': 'limited'},
                3: {'range': 'very limited', 'accuracy': 'many errors', 'complexity': 'very limited'},
                2: {'range': 'very limited', 'accuracy': 'many errors', 'complexity': 'very limited'},
                1: {'range': 'extremely limited', 'accuracy': 'many errors', 'complexity': 'extremely limited'}
            }
        }
    
    def score_essay_realistic(self, essay: str, prompt: str, task_type: str = "Task 2") -> Dict[str, float]:
        """
        Score essay using realistic IELTS criteria
        """
        # Basic validation
        if not essay or len(essay.strip()) < 50:
            return self._get_minimum_scores()
        
        # Analyze essay characteristics
        analysis = self._analyze_essay(essay, prompt, task_type)
        
        # Score each criterion
        task_achievement = self._score_task_achievement(essay, prompt, task_type, analysis)
        coherence_cohesion = self._score_coherence_cohesion(essay, analysis)
        lexical_resource = self._score_lexical_resource(essay, analysis)
        grammatical_range = self._score_grammatical_range(essay, analysis)
        
        # Calculate overall band score
        overall_band_score = (task_achievement + coherence_cohesion + lexical_resource + grammatical_range) / 4
        overall_band_score = round(overall_band_score * 2) / 2  # Round to nearest 0.5
        
        return {
            'task_achievement': task_achievement,
            'coherence_cohesion': coherence_cohesion,
            'lexical_resource': lexical_resource,
            'grammatical_range': grammatical_range,
            'overall_band_score': overall_band_score
        }
    
    def _analyze_essay(self, essay: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """Comprehensive essay analysis"""
        words = simple_word_tokenize(essay.lower())
        sentences = simple_sent_tokenize(essay)
        
        # Basic metrics
        word_count = len([w for w in words if w.isalpha()])
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Grammar analysis (simplified without POS tagging)
        grammar_errors = self._count_grammar_errors(essay, words)
        
        # Vocabulary analysis
        unique_words = len(set([w for w in words if w.isalpha() and w not in self.stop_words]))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Cohesion analysis
        cohesion_markers = self._count_cohesion_markers(essay)
        
        # Task-specific analysis
        task_relevance = self._analyze_task_relevance(essay, prompt, task_type)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'paragraph_count': paragraph_count,
            'grammar_errors': grammar_errors,
            'lexical_diversity': lexical_diversity,
            'cohesion_markers': cohesion_markers,
            'task_relevance': task_relevance,
            'words': words,
            'sentences': sentences
        }
    
    def _score_task_achievement(self, essay: str, prompt: str, task_type: str, analysis: Dict) -> float:
        """Score Task Achievement based on official IELTS criteria"""
        word_count = analysis['word_count']
        task_relevance = analysis['task_relevance']
        
        # Strict word count penalties
        if task_type == "Task 2":
            if word_count < 250:
                base_score = 4.0  # Automatic penalty for under 250 words
            elif word_count < 300:
                base_score = 5.0
            else:
                base_score = 6.0
        else:  # Task 1
            if word_count < 150:
                base_score = 4.0  # Automatic penalty for under 150 words
            elif word_count < 200:
                base_score = 5.0
            else:
                base_score = 6.0
        
        # Task relevance adjustments
        if task_relevance['relevance_score'] < 0.3:
            base_score = max(1.0, base_score - 2.0)  # Severe penalty for off-topic
        elif task_relevance['relevance_score'] < 0.5:
            base_score = max(1.0, base_score - 1.0)  # Moderate penalty
        elif task_relevance['relevance_score'] > 0.8:
            base_score = min(9.0, base_score + 1.0)  # Bonus for high relevance
        
        # Position clarity (for Task 2)
        if task_type == "Task 2":
            position_clarity = self._assess_position_clarity(essay, prompt)
            if position_clarity < 0.3:
                base_score = max(1.0, base_score - 1.0)
            elif position_clarity > 0.7:
                base_score = min(9.0, base_score + 0.5)
        
        return round(max(1.0, min(9.0, base_score)) * 2) / 2
    
    def _score_coherence_cohesion(self, essay: str, analysis: Dict) -> float:
        """Score Coherence and Cohesion"""
        paragraph_count = analysis['paragraph_count']
        cohesion_markers = analysis['cohesion_markers']
        avg_sentence_length = analysis['avg_sentence_length']
        
        # Base score from paragraph structure
        if paragraph_count < 2:
            base_score = 3.0  # Severe penalty for no paragraph structure
        elif paragraph_count < 3:
            base_score = 4.0
        elif paragraph_count < 4:
            base_score = 5.0
        else:
            base_score = 6.0
        
        # Cohesion marker analysis
        cohesion_score = cohesion_markers / analysis['sentence_count'] if analysis['sentence_count'] > 0 else 0
        if cohesion_score < 0.1:
            base_score = max(1.0, base_score - 1.0)  # Penalty for poor cohesion
        elif cohesion_score > 0.3:
            base_score = min(9.0, base_score + 0.5)  # Bonus for good cohesion
        
        # Sentence length variety
        if avg_sentence_length < 10:
            base_score = max(1.0, base_score - 0.5)  # Penalty for very short sentences
        elif avg_sentence_length > 25:
            base_score = max(1.0, base_score - 0.5)  # Penalty for very long sentences
        
        return round(max(1.0, min(9.0, base_score)) * 2) / 2
    
    def _score_lexical_resource(self, essay: str, analysis: Dict) -> float:
        """Score Lexical Resource"""
        lexical_diversity = analysis['lexical_diversity']
        word_count = analysis['word_count']
        
        # Base score from lexical diversity
        if lexical_diversity < 0.3:
            base_score = 3.0  # Very limited vocabulary
        elif lexical_diversity < 0.4:
            base_score = 4.0
        elif lexical_diversity < 0.5:
            base_score = 5.0
        elif lexical_diversity < 0.6:
            base_score = 6.0
        elif lexical_diversity < 0.7:
            base_score = 7.0
        else:
            base_score = 8.0
        
        # Word repetition penalty
        words = analysis['words']
        word_freq = Counter(words)
        repetition_penalty = 0
        for word, count in word_freq.items():
            if count > word_count * 0.05:  # Word appears more than 5% of the time
                repetition_penalty += 0.5
        
        base_score = max(1.0, base_score - repetition_penalty)
        
        # Advanced vocabulary bonus
        advanced_words = self._count_advanced_vocabulary(essay)
        if advanced_words > word_count * 0.1:  # More than 10% advanced vocabulary
            base_score = min(9.0, base_score + 0.5)
        
        return round(max(1.0, min(9.0, base_score)) * 2) / 2
    
    def _score_grammatical_range(self, essay: str, analysis: Dict) -> float:
        """Score Grammatical Range and Accuracy"""
        grammar_errors = analysis['grammar_errors']
        word_count = analysis['word_count']
        sentence_count = analysis['sentence_count']
        
        # Base score from error rate
        error_rate = grammar_errors / word_count if word_count > 0 else 1.0
        
        if error_rate > 0.1:  # More than 10% errors
            base_score = 3.0
        elif error_rate > 0.05:  # More than 5% errors
            base_score = 4.0
        elif error_rate > 0.03:  # More than 3% errors
            base_score = 5.0
        elif error_rate > 0.02:  # More than 2% errors
            base_score = 6.0
        elif error_rate > 0.01:  # More than 1% errors
            base_score = 7.0
        else:
            base_score = 8.0
        
        # Sentence variety bonus
        sentence_variety = self._assess_sentence_variety(essay, analysis['sentences'])
        if sentence_variety > 0.7:
            base_score = min(9.0, base_score + 0.5)
        elif sentence_variety < 0.3:
            base_score = max(1.0, base_score - 0.5)
        
        return round(max(1.0, min(9.0, base_score)) * 2) / 2
    
    def _count_grammar_errors(self, essay: str, words: List[str]) -> int:
        """Count basic grammar errors"""
        errors = 0
        
        # Subject-verb agreement errors (simplified)
        for i in range(len(words) - 1):
            if words[i] in ['he', 'she', 'it'] and words[i+1] in ['are', 'were', 'have']:
                errors += 1
            elif words[i] in ['i', 'you', 'we', 'they'] and words[i+1] in ['is', 'was', 'has']:
                errors += 1
        
        # Article errors (simplified)
        for i in range(len(words) - 1):
            if words[i] in ['a', 'an', 'the'] and words[i+1] in ['a', 'an', 'the']:
                errors += 1
        
        # Double negatives
        if 'not' in words and 'no' in words:
            errors += 1
        
        return errors
    
    def _count_cohesion_markers(self, essay: str) -> int:
        """Count cohesion markers"""
        markers = [
            'however', 'therefore', 'moreover', 'furthermore', 'in addition',
            'consequently', 'as a result', 'on the other hand', 'nevertheless',
            'meanwhile', 'subsequently', 'for example', 'for instance',
            'first', 'second', 'third', 'finally', 'in conclusion',
            'to begin with', 'what is more', 'besides', 'also'
        ]
        
        count = 0
        essay_lower = essay.lower()
        for marker in markers:
            count += essay_lower.count(marker)
        
        return count
    
    def _analyze_task_relevance(self, essay: str, prompt: str, task_type: str) -> Dict[str, float]:
        """Analyze how well the essay addresses the task"""
        # Extract key words from prompt
        prompt_words = set(simple_word_tokenize(prompt.lower()))
        essay_words = set(simple_word_tokenize(essay.lower()))
        
        # Calculate word overlap
        overlap = len(prompt_words.intersection(essay_words))
        relevance_score = overlap / len(prompt_words) if len(prompt_words) > 0 else 0
        
        return {
            'relevance_score': relevance_score,
            'word_overlap': overlap,
            'prompt_words': len(prompt_words)
        }
    
    def _assess_position_clarity(self, essay: str, prompt: str) -> float:
        """Assess clarity of position in Task 2 essays"""
        # Look for opinion indicators
        opinion_words = ['believe', 'think', 'agree', 'disagree', 'support', 'oppose', 'favor', 'prefer']
        essay_lower = essay.lower()
        
        opinion_count = sum(essay_lower.count(word) for word in opinion_words)
        
        # Look for clear position statements
        position_indicators = ['i believe', 'i think', 'in my opinion', 'i agree', 'i disagree']
        position_count = sum(essay_lower.count(phrase) for phrase in position_indicators)
        
        # Calculate clarity score
        total_indicators = opinion_count + position_count
        clarity_score = min(1.0, total_indicators / 5.0)  # Normalize to 0-1
        
        return clarity_score
    
    def _count_advanced_vocabulary(self, essay: str) -> int:
        """Count advanced vocabulary words"""
        # Simple heuristic: words longer than 8 characters that aren't common
        words = simple_word_tokenize(essay.lower())
        advanced_words = [
            'consequently', 'furthermore', 'nevertheless', 'subsequently',
            'comprehensive', 'significant', 'substantial', 'considerable',
            'demonstrate', 'illustrate', 'emphasize', 'acknowledge',
            'contemporary', 'sophisticated', 'fundamental', 'essential'
        ]
        
        count = 0
        for word in words:
            if len(word) > 8 and word not in self.stop_words:
                count += 1
            if word in advanced_words:
                count += 1
        
        return count
    
    def _assess_sentence_variety(self, essay: str, sentences: List[str]) -> float:
        """Assess sentence variety"""
        if len(sentences) < 2:
            return 0.0
        
        # Calculate sentence length variety
        lengths = [len(sentence.split()) for sentence in sentences]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        
        # Normalize variance (higher variance = more variety)
        variety_score = min(1.0, length_variance / 100.0)
        
        return variety_score
    
    def _get_minimum_scores(self) -> Dict[str, float]:
        """Return minimum scores for invalid essays"""
        return {
            'task_achievement': 1.0,
            'coherence_cohesion': 1.0,
            'lexical_resource': 1.0,
            'grammatical_range': 1.0,
            'overall_band_score': 1.0
        }
