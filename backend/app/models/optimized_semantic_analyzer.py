"""
Optimized Semantic Analyzer
Fast, production-ready semantic analysis without NLTK performance issues
"""

import re
from typing import Dict, List, Any
import numpy as np
from collections import Counter

class OptimizedSemanticAnalyzer:
    """
    Fast semantic analysis optimized for production use
    Uses regex and efficient algorithms instead of heavy NLTK operations
    """
    
    def __init__(self):
        # Pre-compiled regex patterns for speed
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r'\b\w+\b')
        
        # IELTS-specific vocabulary levels (pre-loaded for speed)
        self.basic_vocab = {
            'good', 'bad', 'big', 'small', 'happy', 'sad', 'easy', 'hard', 'new', 'old',
            'important', 'different', 'same', 'first', 'last', 'next', 'best', 'worst',
            'think', 'know', 'want', 'need', 'like', 'love', 'hate', 'see', 'hear', 'feel'
        }
        
        self.advanced_vocab = {
            'significant', 'considerable', 'substantial', 'essential', 'crucial', 'vital',
            'demonstrate', 'illustrate', 'emphasize', 'acknowledge', 'recognize', 'realize',
            'consequently', 'furthermore', 'nevertheless', 'subsequently', 'meanwhile',
            'contemporary', 'sophisticated', 'fundamental', 'comprehensive', 'extensive',
            'paradigm', 'methodology', 'phenomenon', 'hypothesis', 'empirical', 'theoretical'
        }
        
        # Cohesive devices
        self.cohesive_devices = {
            'addition': ['furthermore', 'moreover', 'in addition', 'besides', 'also', 'additionally'],
            'contrast': ['however', 'nevertheless', 'on the other hand', 'in contrast', 'whereas', 'while'],
            'cause_effect': ['therefore', 'consequently', 'as a result', 'thus', 'hence', 'because of this'],
            'example': ['for example', 'for instance', 'such as', 'namely', 'specifically'],
            'conclusion': ['in conclusion', 'to conclude', 'to sum up', 'overall', 'in summary'],
            'time': ['firstly', 'secondly', 'finally', 'subsequently', 'meanwhile', 'initially']
        }
        
        # Academic vocabulary
        self.academic_vocab = {
            'analysis', 'approach', 'area', 'assessment', 'assume', 'authority', 'available',
            'benefit', 'concept', 'consistent', 'constitutional', 'context', 'contract', 'create',
            'data', 'definition', 'derived', 'distribution', 'economic', 'environment', 'established',
            'estimate', 'evidence', 'export', 'factors', 'financial', 'formula', 'function',
            'identified', 'income', 'indicate', 'individual', 'interpretation', 'involved', 'issues',
            'labour', 'legal', 'legislation', 'major', 'method', 'occur', 'percent', 'period',
            'policy', 'principle', 'procedure', 'process', 'required', 'research', 'response',
            'role', 'section', 'sector', 'significant', 'similar', 'source', 'specific', 'structure',
            'theory', 'variables'
        }
    
    def analyze_essay_fast(self, essay: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Fast semantic analysis optimized for production
        """
        # Basic text processing (fast)
        sentences = self._fast_sentence_split(essay)
        words = self._fast_word_tokenize(essay.lower())
        
        # Core analysis
        analysis = {
            'basic_metrics': self._analyze_basic_metrics_fast(essay, sentences, words),
            'vocabulary_analysis': self._analyze_vocabulary_fast(words),
            'coherence_analysis': self._analyze_coherence_fast(essay, sentences),
            'grammar_analysis': self._analyze_grammar_fast(essay, words),
            'task_relevance': self._analyze_task_relevance_fast(essay, prompt, words),
            'academic_language': self._analyze_academic_language_fast(words),
            'semantic_coherence': self._analyze_semantic_coherence_fast(sentences, words)
        }
        
        return analysis
    
    def _fast_sentence_split(self, text: str) -> List[str]:
        """Fast sentence splitting using regex"""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _fast_word_tokenize(self, text: str) -> List[str]:
        """Fast word tokenization using regex"""
        return self.word_pattern.findall(text)
    
    def _analyze_basic_metrics_fast(self, essay: str, sentences: List[str], words: List[str]) -> Dict[str, Any]:
        """Fast basic metrics analysis"""
        word_count = len([w for w in words if w.isalpha()])
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Word length analysis
        word_lengths = [len(w) for w in words if w.isalpha()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'sentence_length_variance': np.var([len(s.split()) for s in sentences]) if sentences else 0
        }
    
    def _analyze_vocabulary_fast(self, words: List[str]) -> Dict[str, Any]:
        """Fast vocabulary analysis"""
        content_words = [w for w in words if w.isalpha() and len(w) > 2]
        
        if not content_words:
            return {
                'lexical_diversity': 0.0,
                'vocabulary_level': 'basic',
                'academic_ratio': 0.0,
                'repetition_score': 0.0
            }
        
        # Vocabulary level analysis
        basic_count = sum(1 for w in content_words if w in self.basic_vocab)
        advanced_count = sum(1 for w in content_words if w in self.advanced_vocab)
        academic_count = sum(1 for w in content_words if w in self.academic_vocab)
        
        # Lexical diversity
        unique_words = len(set(content_words))
        lexical_diversity = unique_words / len(content_words)
        
        # Repetition analysis
        word_freq = Counter(content_words)
        repetition_penalty = sum(1 for word, freq in word_freq.items() if freq > len(content_words) * 0.05)
        
        # Determine vocabulary level
        if advanced_count > len(content_words) * 0.15:
            vocab_level = 'advanced'
        elif advanced_count > len(content_words) * 0.05:
            vocab_level = 'intermediate'
        else:
            vocab_level = 'basic'
        
        return {
            'lexical_diversity': lexical_diversity,
            'vocabulary_level': vocab_level,
            'basic_ratio': basic_count / len(content_words),
            'advanced_ratio': advanced_count / len(content_words),
            'academic_ratio': academic_count / len(content_words),
            'repetition_score': 1.0 - (repetition_penalty / len(content_words)),
            'unique_words': unique_words,
            'total_content_words': len(content_words)
        }
    
    def _analyze_coherence_fast(self, essay: str, sentences: List[str]) -> Dict[str, Any]:
        """Fast coherence analysis"""
        essay_lower = essay.lower()
        
        # Discourse markers analysis
        discourse_analysis = {}
        total_markers = 0
        
        for category, markers in self.cohesive_devices.items():
            count = sum(essay_lower.count(marker) for marker in markers)
            discourse_analysis[category] = count
            total_markers += count
        
        # Discourse marker variety
        used_categories = sum(1 for count in discourse_analysis.values() if count > 0)
        variety_score = used_categories / len(self.cohesive_devices)
        
        # Paragraph structure analysis
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        structure_score = min(1.0, len(paragraphs) / 4.0)  # Ideal: 4 paragraphs
        
        return {
            'discourse_markers': discourse_analysis,
            'total_markers': total_markers,
            'variety_score': variety_score,
            'marker_density': total_markers / len(sentences) if sentences else 0,
            'structure_score': structure_score,
            'paragraph_count': len(paragraphs)
        }
    
    def _analyze_grammar_fast(self, essay: str, words: List[str]) -> Dict[str, Any]:
        """Fast grammar analysis"""
        # Basic grammar error detection
        grammar_errors = 0
        
        # Subject-verb agreement (simplified)
        for i in range(len(words) - 1):
            if words[i] in ['he', 'she', 'it'] and words[i+1] in ['are', 'were', 'have']:
                grammar_errors += 1
            elif words[i] in ['i', 'you', 'we', 'they'] and words[i+1] in ['is', 'was', 'has']:
                grammar_errors += 1
        
        # Sentence complexity analysis
        sentences = self._fast_sentence_split(essay)
        complex_sentences = 0
        compound_sentences = 0
        
        for sentence in sentences:
            if any(marker in sentence.lower() for marker in ['because', 'although', 'while', 'since', 'if', 'unless', 'when', 'where', 'that', 'which', 'who']):
                complex_sentences += 1
            elif any(marker in sentence.lower() for marker in ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']):
                compound_sentences += 1
        
        total_sentences = len(sentences)
        complexity_score = (complex_sentences + compound_sentences) / total_sentences if total_sentences > 0 else 0
        
        # Error rate
        error_rate = grammar_errors / len(words) if words else 0
        
        return {
            'grammar_errors': grammar_errors,
            'error_rate': error_rate,
            'complex_sentences': complex_sentences,
            'compound_sentences': compound_sentences,
            'complexity_score': complexity_score,
            'sentence_variety_score': min(1.0, complexity_score * 2)
        }
    
    def _analyze_task_relevance_fast(self, essay: str, prompt: str, words: List[str]) -> Dict[str, Any]:
        """Fast task relevance analysis"""
        # Extract key concepts from prompt and essay
        prompt_words = set(self._fast_word_tokenize(prompt.lower()))
        essay_words = set(words)
        
        # Calculate overlap
        overlap = len(prompt_words.intersection(essay_words))
        relevance_score = overlap / len(prompt_words) if len(prompt_words) > 0 else 0
        
        # Position clarity for Task 2
        position_clarity = 0
        if 'task 2' in prompt.lower() or 'discuss' in prompt.lower() or 'opinion' in prompt.lower():
            opinion_words = ['believe', 'think', 'agree', 'disagree', 'support', 'oppose', 'favor', 'prefer']
            opinion_count = sum(essay.lower().count(word) for word in opinion_words)
            position_clarity = min(1.0, opinion_count / 3.0)  # Normalize to 0-1
        
        return {
            'relevance_score': relevance_score,
            'position_clarity': position_clarity,
            'concept_overlap': overlap,
            'prompt_words': len(prompt_words),
            'essay_words': len(essay_words)
        }
    
    def _analyze_academic_language_fast(self, words: List[str]) -> Dict[str, Any]:
        """Fast academic language analysis"""
        # Academic vocabulary usage
        academic_count = sum(1 for w in words if w in self.academic_vocab)
        academic_ratio = academic_count / len(words) if words else 0
        
        # Formal language markers
        formal_markers = ['furthermore', 'moreover', 'consequently', 'therefore', 'however', 'nevertheless']
        formal_count = sum(1 for w in words if w in formal_markers)
        formality_score = formal_count / len(words) if words else 0
        
        # Nominalization (words ending in -tion, -sion, -ment, -ness, -ity)
        nominalization_patterns = ['tion', 'sion', 'ment', 'ness', 'ity', 'ty']
        nominalized_words = sum(1 for word in words if any(pattern in word for pattern in nominalization_patterns))
        nominalization_score = nominalized_words / len(words) if words else 0
        
        return {
            'academic_ratio': academic_ratio,
            'formality_score': formality_score,
            'nominalization_score': nominalization_score,
            'academic_language_score': (academic_ratio + formality_score + nominalization_score) / 3
        }
    
    def _analyze_semantic_coherence_fast(self, sentences: List[str], words: List[str]) -> Dict[str, Any]:
        """Fast semantic coherence analysis"""
        if len(sentences) < 2:
            return {
                'topic_consistency': 1.0,
                'sentence_similarity': 1.0,
                'lexical_cohesion': 1.0
            }
        
        # Topic consistency (simplified)
        sentence_keywords = []
        for sentence in sentences:
            sentence_words = self._fast_word_tokenize(sentence.lower())
            # Get content words (longer than 2 characters, not common words)
            content_words = [w for w in sentence_words if len(w) > 2 and w not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}]
            sentence_keywords.append(set(content_words))
        
        # Calculate consistency between adjacent sentences
        consistency_scores = []
        for i in range(len(sentence_keywords) - 1):
            if sentence_keywords[i] and sentence_keywords[i + 1]:
                overlap = len(sentence_keywords[i].intersection(sentence_keywords[i + 1]))
                union = len(sentence_keywords[i].union(sentence_keywords[i + 1]))
                consistency = overlap / union if union > 0 else 0
                consistency_scores.append(consistency)
        
        topic_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Lexical cohesion (word repetition)
        word_freq = Counter(words)
        lexical_cohesion = sum(1 for word, freq in word_freq.items() if freq > 1) / len(word_freq) if word_freq else 0
        
        return {
            'topic_consistency': topic_consistency,
            'sentence_similarity': topic_consistency,  # Simplified
            'lexical_cohesion': lexical_cohesion
        }
