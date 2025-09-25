"""
Advanced Semantic Analyzer for IELTS Essays
Uses NLTK, spaCy, and advanced NLP techniques for deep analysis
"""

import re
import nltk
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import Counter
import math

# NLTK imports
from nltk.tokenize import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# Text statistics
from textstat import flesch_reading_ease, gunning_fog, smog_index, coleman_liau_index

class AdvancedSemanticAnalyzer:
    """
    Advanced semantic analysis for IELTS essays using NLTK and linguistic features
    """
    
    def __init__(self):
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # IELTS-specific vocabulary levels
        self.basic_words = self._load_basic_vocabulary()
        self.intermediate_words = self._load_intermediate_vocabulary()
        self.advanced_words = self._load_advanced_vocabulary()
        
        # Cohesive devices
        self.cohesive_devices = {
            'addition': ['furthermore', 'moreover', 'in addition', 'besides', 'also', 'additionally'],
            'contrast': ['however', 'nevertheless', 'on the other hand', 'in contrast', 'whereas', 'while'],
            'cause_effect': ['therefore', 'consequently', 'as a result', 'thus', 'hence', 'because of this'],
            'example': ['for example', 'for instance', 'such as', 'namely', 'specifically'],
            'conclusion': ['in conclusion', 'to conclude', 'to sum up', 'overall', 'in summary'],
            'time': ['firstly', 'secondly', 'finally', 'subsequently', 'meanwhile', 'initially']
        }
        
        # Academic vocabulary (AWL - Academic Word List)
        self.academic_vocabulary = self._load_academic_vocabulary()
        
    def analyze_essay_semantics(self, essay: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of the essay
        """
        # Basic text processing
        sentences = sent_tokenize(essay)
        words = word_tokenize(essay.lower())
        pos_tags = pos_tag(words)
        
        # Semantic analysis components
        analysis = {
            'basic_metrics': self._analyze_basic_metrics(essay, sentences, words),
            'semantic_coherence': self._analyze_semantic_coherence(essay, sentences, words),
            'vocabulary_sophistication': self._analyze_vocabulary_sophistication(words, pos_tags),
            'grammatical_complexity': self._analyze_grammatical_complexity(sentences, pos_tags),
            'discourse_markers': self._analyze_discourse_markers(essay),
            'sentiment_analysis': self._analyze_sentiment(essay),
            'readability_metrics': self._analyze_readability(essay),
            'task_relevance': self._analyze_task_relevance_semantic(essay, prompt, task_type),
            'cohesive_devices': self._analyze_cohesive_devices(essay),
            'academic_language': self._analyze_academic_language(words),
            'syntactic_complexity': self._analyze_syntactic_complexity(sentences, pos_tags)
        }
        
        return analysis
    
    def _analyze_basic_metrics(self, essay: str, sentences: List[str], words: List[str]) -> Dict[str, Any]:
        """Basic text metrics"""
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
    
    def _analyze_semantic_coherence(self, essay: str, sentences: List[str], words: List[str]) -> Dict[str, Any]:
        """Analyze semantic coherence and topic consistency"""
        # Topic modeling (simplified)
        word_freq = Counter([w for w in words if w.isalpha() and w not in self.stop_words])
        top_words = [word for word, freq in word_freq.most_common(10)]
        
        # Semantic similarity between sentences (simplified)
        sentence_similarity = self._calculate_sentence_similarity(sentences)
        
        # Topic drift analysis
        topic_consistency = self._analyze_topic_consistency(sentences)
        
        return {
            'top_words': top_words,
            'sentence_similarity': sentence_similarity,
            'topic_consistency': topic_consistency,
            'lexical_cohesion': self._calculate_lexical_cohesion(words)
        }
    
    def _analyze_vocabulary_sophistication(self, words: List[str], pos_tags: List[Tuple]) -> Dict[str, Any]:
        """Analyze vocabulary sophistication and lexical diversity"""
        # Remove stopwords and punctuation
        content_words = [w for w, pos in pos_tags if w.isalpha() and w not in self.stop_words]
        
        # Vocabulary level analysis
        basic_count = sum(1 for w in content_words if w in self.basic_words)
        intermediate_count = sum(1 for w in content_words if w in self.intermediate_words)
        advanced_count = sum(1 for w in content_words if w in self.advanced_words)
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = len(set(content_words))
        total_words = len(content_words)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Academic vocabulary usage
        academic_count = sum(1 for w in content_words if w in self.academic_vocabulary)
        academic_ratio = academic_count / total_words if total_words > 0 else 0
        
        # Word frequency analysis
        word_freq = Counter(content_words)
        repetition_penalty = sum(1 for word, freq in word_freq.items() if freq > total_words * 0.05)
        
        return {
            'basic_vocabulary_ratio': basic_count / total_words if total_words > 0 else 0,
            'intermediate_vocabulary_ratio': intermediate_count / total_words if total_words > 0 else 0,
            'advanced_vocabulary_ratio': advanced_count / total_words if total_words > 0 else 0,
            'lexical_diversity': lexical_diversity,
            'academic_vocabulary_ratio': academic_ratio,
            'repetition_penalty': repetition_penalty,
            'unique_words': unique_words,
            'total_content_words': total_words
        }
    
    def _analyze_grammatical_complexity(self, sentences: List[str], pos_tags: List[Tuple]) -> Dict[str, Any]:
        """Analyze grammatical complexity and accuracy"""
        # Sentence complexity analysis
        complex_sentences = 0
        compound_sentences = 0
        simple_sentences = 0
        
        for sentence in sentences:
            if self._is_complex_sentence(sentence):
                complex_sentences += 1
            elif self._is_compound_sentence(sentence):
                compound_sentences += 1
            else:
                simple_sentences += 1
        
        # Grammar error detection
        grammar_errors = self._detect_grammar_errors(pos_tags)
        
        # Verb tense analysis
        tense_consistency = self._analyze_tense_consistency(pos_tags)
        
        # Passive voice usage
        passive_voice_ratio = self._analyze_passive_voice(pos_tags)
        
        return {
            'complex_sentences': complex_sentences,
            'compound_sentences': compound_sentences,
            'simple_sentences': simple_sentences,
            'grammar_errors': grammar_errors,
            'tense_consistency': tense_consistency,
            'passive_voice_ratio': passive_voice_ratio,
            'sentence_variety_score': self._calculate_sentence_variety_score(complex_sentences, compound_sentences, simple_sentences)
        }
    
    def _analyze_discourse_markers(self, essay: str) -> Dict[str, Any]:
        """Analyze discourse markers and cohesive devices"""
        essay_lower = essay.lower()
        
        discourse_analysis = {}
        total_markers = 0
        
        for category, markers in self.cohesive_devices.items():
            count = sum(essay_lower.count(marker) for marker in markers)
            discourse_analysis[category] = count
            total_markers += count
        
        # Discourse marker variety
        used_categories = sum(1 for count in discourse_analysis.values() if count > 0)
        variety_score = used_categories / len(self.cohesive_devices)
        
        return {
            'discourse_markers': discourse_analysis,
            'total_markers': total_markers,
            'variety_score': variety_score,
            'marker_density': total_markers / len(sent_tokenize(essay)) if len(sent_tokenize(essay)) > 0 else 0
        }
    
    def _analyze_sentiment(self, essay: str) -> Dict[str, Any]:
        """Analyze sentiment and emotional tone"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(essay)
        
        # Academic tone analysis
        academic_tone_score = self._analyze_academic_tone(essay)
        
        return {
            'sentiment_scores': sentiment_scores,
            'academic_tone_score': academic_tone_score,
            'emotional_appropriateness': self._assess_emotional_appropriateness(sentiment_scores)
        }
    
    def _analyze_readability(self, essay: str) -> Dict[str, Any]:
        """Analyze readability and text complexity"""
        try:
            flesch_score = flesch_reading_ease(essay)
            fog_index = gunning_fog(essay)
            smog_score = smog_index(essay)
            coleman_score = coleman_liau_index(essay)
        except:
            flesch_score = fog_index = smog_score = coleman_score = 0
        
        return {
            'flesch_reading_ease': flesch_score,
            'gunning_fog_index': fog_index,
            'smog_index': smog_score,
            'coleman_liau_index': coleman_score,
            'readability_level': self._determine_readability_level(flesch_score)
        }
    
    def _analyze_task_relevance_semantic(self, essay: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """Advanced task relevance analysis using semantic similarity"""
        # Extract key concepts from prompt
        prompt_concepts = self._extract_key_concepts(prompt)
        essay_concepts = self._extract_key_concepts(essay)
        
        # Calculate semantic overlap
        concept_overlap = len(prompt_concepts.intersection(essay_concepts))
        relevance_score = concept_overlap / len(prompt_concepts) if len(prompt_concepts) > 0 else 0
        
        # Position clarity for Task 2
        position_clarity = 0
        if task_type == "Task 2":
            position_clarity = self._analyze_position_clarity(essay, prompt)
        
        # Argument development
        argument_development = self._analyze_argument_development(essay, task_type)
        
        return {
            'concept_overlap': concept_overlap,
            'relevance_score': relevance_score,
            'position_clarity': position_clarity,
            'argument_development': argument_development,
            'prompt_concepts': list(prompt_concepts),
            'essay_concepts': list(essay_concepts)
        }
    
    def _analyze_cohesive_devices(self, essay: str) -> Dict[str, Any]:
        """Analyze cohesive devices and text flow"""
        essay_lower = essay.lower()
        
        # Reference analysis (pronouns, demonstratives)
        reference_devices = self._analyze_reference_devices(essay)
        
        # Substitution and ellipsis
        substitution_devices = self._analyze_substitution_devices(essay)
        
        # Conjunction analysis
        conjunction_analysis = self._analyze_conjunctions(essay)
        
        return {
            'reference_devices': reference_devices,
            'substitution_devices': substitution_devices,
            'conjunction_analysis': conjunction_analysis,
            'cohesion_score': self._calculate_overall_cohesion_score(reference_devices, substitution_devices, conjunction_analysis)
        }
    
    def _analyze_academic_language(self, words: List[str]) -> Dict[str, Any]:
        """Analyze academic language features"""
        # Nominalization (turning verbs into nouns)
        nominalization_score = self._analyze_nominalization(words)
        
        # Hedging language (modals, tentative language)
        hedging_score = self._analyze_hedging_language(words)
        
        # Formal language markers
        formality_score = self._analyze_formality_markers(words)
        
        return {
            'nominalization_score': nominalization_score,
            'hedging_score': hedging_score,
            'formality_score': formality_score,
            'academic_language_score': (nominalization_score + hedging_score + formality_score) / 3
        }
    
    def _analyze_syntactic_complexity(self, sentences: List[str], pos_tags: List[Tuple]) -> Dict[str, Any]:
        """Analyze syntactic complexity"""
        # Subordination ratio
        subordination_ratio = self._calculate_subordination_ratio(sentences)
        
        # Coordination ratio
        coordination_ratio = self._calculate_coordination_ratio(sentences)
        
        # Phrase complexity
        phrase_complexity = self._analyze_phrase_complexity(pos_tags)
        
        return {
            'subordination_ratio': subordination_ratio,
            'coordination_ratio': coordination_ratio,
            'phrase_complexity': phrase_complexity,
            'syntactic_complexity_score': (subordination_ratio + coordination_ratio + phrase_complexity) / 3
        }
    
    # Helper methods
    def _load_basic_vocabulary(self) -> set:
        """Load basic vocabulary list"""
        return {
            'good', 'bad', 'big', 'small', 'happy', 'sad', 'easy', 'hard', 'new', 'old',
            'important', 'different', 'same', 'first', 'last', 'next', 'best', 'worst',
            'think', 'know', 'want', 'need', 'like', 'love', 'hate', 'see', 'hear', 'feel'
        }
    
    def _load_intermediate_vocabulary(self) -> set:
        """Load intermediate vocabulary list"""
        return {
            'significant', 'considerable', 'substantial', 'essential', 'crucial', 'vital',
            'demonstrate', 'illustrate', 'emphasize', 'acknowledge', 'recognize', 'realize',
            'consequently', 'furthermore', 'nevertheless', 'subsequently', 'meanwhile',
            'contemporary', 'sophisticated', 'fundamental', 'comprehensive', 'extensive'
        }
    
    def _load_advanced_vocabulary(self) -> set:
        """Load advanced vocabulary list"""
        return {
            'paradigm', 'methodology', 'phenomenon', 'hypothesis', 'empirical', 'theoretical',
            'sophisticated', 'comprehensive', 'multifaceted', 'interdisciplinary', 'paradoxical',
            'consequently', 'furthermore', 'nevertheless', 'subsequently', 'meanwhile',
            'contemporary', 'sophisticated', 'fundamental', 'comprehensive', 'extensive'
        }
    
    def _load_academic_vocabulary(self) -> set:
        """Load Academic Word List (AWL)"""
        return {
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
    
    def _calculate_sentence_similarity(self, sentences: List[str]) -> float:
        """Calculate semantic similarity between sentences (simplified)"""
        if len(sentences) < 2:
            return 0.0
        
        # Simple word overlap similarity
        similarities = []
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i + 1].lower()))
            
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                similarity = overlap / len(words1.union(words2))
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _analyze_topic_consistency(self, sentences: List[str]) -> float:
        """Analyze topic consistency across sentences"""
        if len(sentences) < 2:
            return 1.0
        
        # Extract key words from each sentence
        sentence_keywords = []
        for sentence in sentences:
            words = [w for w in word_tokenize(sentence.lower()) if w.isalpha() and w not in self.stop_words]
            sentence_keywords.append(set(words))
        
        # Calculate consistency
        consistency_scores = []
        for i in range(len(sentence_keywords) - 1):
            if sentence_keywords[i] and sentence_keywords[i + 1]:
                overlap = len(sentence_keywords[i].intersection(sentence_keywords[i + 1]))
                union = len(sentence_keywords[i].union(sentence_keywords[i + 1]))
                consistency = overlap / union if union > 0 else 0
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_lexical_cohesion(self, words: List[str]) -> float:
        """Calculate lexical cohesion score"""
        content_words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        if len(content_words) < 2:
            return 0.0
        
        # Calculate lexical chains (simplified)
        lexical_chains = 0
        for i in range(len(content_words) - 1):
            if content_words[i] == content_words[i + 1]:
                lexical_chains += 1
        
        return lexical_chains / len(content_words)
    
    def _is_complex_sentence(self, sentence: str) -> bool:
        """Check if sentence is complex (has subordination)"""
        complex_markers = ['because', 'although', 'while', 'since', 'if', 'unless', 'when', 'where', 'that', 'which', 'who']
        return any(marker in sentence.lower() for marker in complex_markers)
    
    def _is_compound_sentence(self, sentence: str) -> bool:
        """Check if sentence is compound (has coordination)"""
        compound_markers = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']
        return any(marker in sentence.lower() for marker in compound_markers)
    
    def _detect_grammar_errors(self, pos_tags: List[Tuple]) -> int:
        """Detect basic grammar errors"""
        errors = 0
        words = [tag[0] for tag in pos_tags]
        
        # Subject-verb agreement
        for i in range(len(words) - 1):
            if words[i] in ['he', 'she', 'it'] and words[i+1] in ['are', 'were', 'have']:
                errors += 1
            elif words[i] in ['i', 'you', 'we', 'they'] and words[i+1] in ['is', 'was', 'has']:
                errors += 1
        
        return errors
    
    def _analyze_tense_consistency(self, pos_tags: List[Tuple]) -> float:
        """Analyze tense consistency"""
        verbs = [word for word, pos in pos_tags if pos.startswith('V')]
        
        if len(verbs) < 2:
            return 1.0
        
        # Simple tense analysis (simplified)
        past_verbs = sum(1 for word in verbs if word.endswith('ed') or word in ['was', 'were', 'had', 'did'])
        present_verbs = sum(1 for word in verbs if not word.endswith('ed') and word not in ['was', 'were', 'had', 'did'])
        
        # Calculate consistency
        if past_verbs > present_verbs:
            consistency = past_verbs / len(verbs)
        else:
            consistency = present_verbs / len(verbs)
        
        return consistency
    
    def _analyze_passive_voice(self, pos_tags: List[Tuple]) -> float:
        """Analyze passive voice usage"""
        words = [word for word, pos in pos_tags]
        
        passive_markers = ['be', 'am', 'is', 'are', 'was', 'were', 'been', 'being']
        passive_count = 0
        
        for i in range(len(words) - 1):
            if words[i] in passive_markers and words[i+1].endswith('ed'):
                passive_count += 1
        
        return passive_count / len(words) if words else 0.0
    
    def _calculate_sentence_variety_score(self, complex: int, compound: int, simple: int) -> float:
        """Calculate sentence variety score"""
        total = complex + compound + simple
        if total == 0:
            return 0.0
        
        # Ideal distribution: 40% complex, 30% compound, 30% simple
        ideal_complex = 0.4
        ideal_compound = 0.3
        ideal_simple = 0.3
        
        actual_complex = complex / total
        actual_compound = compound / total
        actual_simple = simple / total
        
        # Calculate deviation from ideal
        deviation = abs(actual_complex - ideal_complex) + abs(actual_compound - ideal_compound) + abs(actual_simple - ideal_simple)
        
        return max(0.0, 1.0 - deviation)
    
    def _analyze_academic_tone(self, essay: str) -> float:
        """Analyze academic tone appropriateness"""
        essay_lower = essay.lower()
        
        # Academic tone markers
        academic_markers = ['furthermore', 'moreover', 'consequently', 'therefore', 'however', 'nevertheless']
        academic_count = sum(essay_lower.count(marker) for marker in academic_markers)
        
        # Informal tone markers
        informal_markers = ['gonna', 'wanna', 'gotta', 'yeah', 'ok', 'cool', 'awesome']
        informal_count = sum(essay_lower.count(marker) for marker in informal_markers)
        
        # Calculate tone score
        total_markers = academic_count + informal_count
        if total_markers == 0:
            return 0.5  # Neutral
        
        return academic_count / total_markers
    
    def _assess_emotional_appropriateness(self, sentiment_scores: Dict[str, float]) -> float:
        """Assess if emotional tone is appropriate for academic writing"""
        # Academic writing should be neutral to slightly positive
        compound_score = sentiment_scores.get('compound', 0)
        
        # Ideal range: -0.1 to 0.3
        if -0.1 <= compound_score <= 0.3:
            return 1.0
        elif -0.3 <= compound_score < -0.1 or 0.3 < compound_score <= 0.5:
            return 0.7
        else:
            return 0.3
    
    def _determine_readability_level(self, flesch_score: float) -> str:
        """Determine readability level based on Flesch score"""
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def _extract_key_concepts(self, text: str) -> set:
        """Extract key concepts from text"""
        words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in self.stop_words]
        # Simple concept extraction (in practice, you'd use more sophisticated methods)
        return set(words)
    
    def _analyze_position_clarity(self, essay: str, prompt: str) -> float:
        """Analyze clarity of position in Task 2 essays"""
        essay_lower = essay.lower()
        
        # Look for opinion indicators
        opinion_words = ['believe', 'think', 'agree', 'disagree', 'support', 'oppose', 'favor', 'prefer']
        opinion_count = sum(essay_lower.count(word) for word in opinion_words)
        
        # Look for clear position statements
        position_indicators = ['i believe', 'i think', 'in my opinion', 'i agree', 'i disagree']
        position_count = sum(essay_lower.count(phrase) for phrase in position_indicators)
        
        # Calculate clarity score
        total_indicators = opinion_count + position_count
        return min(1.0, total_indicators / 5.0)
    
    def _analyze_argument_development(self, essay: str, task_type: str) -> float:
        """Analyze argument development"""
        if task_type != "Task 2":
            return 1.0
        
        # Look for argument structure indicators
        structure_markers = ['first', 'second', 'third', 'furthermore', 'moreover', 'in addition', 'however', 'on the other hand']
        essay_lower = essay.lower()
        
        structure_count = sum(essay_lower.count(marker) for marker in structure_markers)
        
        # Look for examples and evidence
        example_markers = ['for example', 'for instance', 'such as', 'specifically', 'in particular']
        example_count = sum(essay_lower.count(marker) for marker in example_markers)
        
        # Calculate development score
        total_markers = structure_count + example_count
        return min(1.0, total_markers / 10.0)
    
    def _analyze_reference_devices(self, essay: str) -> Dict[str, int]:
        """Analyze reference devices"""
        essay_lower = essay.lower()
        
        # Pronouns
        pronouns = ['he', 'she', 'it', 'they', 'this', 'that', 'these', 'those', 'which', 'who']
        pronoun_count = sum(essay_lower.count(pronoun) for pronoun in pronouns)
        
        # Demonstratives
        demonstratives = ['this', 'that', 'these', 'those']
        demonstrative_count = sum(essay_lower.count(demo) for demo in demonstratives)
        
        return {
            'pronouns': pronoun_count,
            'demonstratives': demonstrative_count,
            'total_references': pronoun_count + demonstrative_count
        }
    
    def _analyze_substitution_devices(self, essay: str) -> Dict[str, int]:
        """Analyze substitution and ellipsis devices"""
        essay_lower = essay.lower()
        
        # Substitution markers
        substitution_markers = ['one', 'ones', 'do', 'does', 'did', 'so', 'not']
        substitution_count = sum(essay_lower.count(marker) for marker in substitution_markers)
        
        return {
            'substitution_count': substitution_count
        }
    
    def _analyze_conjunctions(self, essay: str) -> Dict[str, int]:
        """Analyze conjunction usage"""
        essay_lower = essay.lower()
        
        # Coordinating conjunctions
        coordinators = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']
        coordinator_count = sum(essay_lower.count(coord) for coord in coordinators)
        
        # Subordinating conjunctions
        subordinators = ['because', 'although', 'while', 'since', 'if', 'unless', 'when', 'where']
        subordinator_count = sum(essay_lower.count(sub) for sub in subordinators)
        
        return {
            'coordinators': coordinator_count,
            'subordinators': subordinator_count,
            'total_conjunctions': coordinator_count + subordinator_count
        }
    
    def _calculate_overall_cohesion_score(self, reference: Dict, substitution: Dict, conjunction: Dict) -> float:
        """Calculate overall cohesion score"""
        # Weighted combination of cohesion devices
        reference_score = min(1.0, reference['total_references'] / 10.0)
        substitution_score = min(1.0, substitution['substitution_count'] / 5.0)
        conjunction_score = min(1.0, conjunction['total_conjunctions'] / 15.0)
        
        return (reference_score + substitution_score + conjunction_score) / 3.0
    
    def _analyze_nominalization(self, words: List[str]) -> float:
        """Analyze nominalization (turning verbs into nouns)"""
        # Common nominalization patterns
        nominalization_patterns = ['tion', 'sion', 'ment', 'ness', 'ity', 'ty']
        
        nominalized_words = 0
        for word in words:
            if any(pattern in word for pattern in nominalization_patterns):
                nominalized_words += 1
        
        return nominalized_words / len(words) if words else 0.0
    
    def _analyze_hedging_language(self, words: List[str]) -> float:
        """Analyze hedging language (tentative language)"""
        hedging_words = ['might', 'may', 'could', 'possibly', 'perhaps', 'likely', 'probably', 'seems', 'appears']
        
        hedging_count = sum(1 for word in words if word in hedging_words)
        return hedging_count / len(words) if words else 0.0
    
    def _analyze_formality_markers(self, words: List[str]) -> float:
        """Analyze formality markers"""
        formal_markers = ['furthermore', 'moreover', 'consequently', 'therefore', 'however', 'nevertheless']
        
        formal_count = sum(1 for word in words if word in formal_markers)
        return formal_count / len(words) if words else 0.0
    
    def _calculate_subordination_ratio(self, sentences: List[str]) -> float:
        """Calculate subordination ratio"""
        if not sentences:
            return 0.0
        
        subordinate_clauses = 0
        total_clauses = 0
        
        for sentence in sentences:
            # Count subordinate conjunctions
            subordinating_conjunctions = ['because', 'although', 'while', 'since', 'if', 'unless', 'when', 'where', 'that', 'which', 'who']
            for conj in subordinating_conjunctions:
                subordinate_clauses += sentence.lower().count(conj)
            
            # Estimate total clauses (simplified)
            total_clauses += sentence.count(',') + 1
        
        return subordinate_clauses / total_clauses if total_clauses > 0 else 0.0
    
    def _calculate_coordination_ratio(self, sentences: List[str]) -> float:
        """Calculate coordination ratio"""
        if not sentences:
            return 0.0
        
        coordinating_conjunctions = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']
        total_coordinators = 0
        total_sentences = len(sentences)
        
        for sentence in sentences:
            for coord in coordinating_conjunctions:
                total_coordinators += sentence.lower().count(coord)
        
        return total_coordinators / total_sentences if total_sentences > 0 else 0.0
    
    def _analyze_phrase_complexity(self, pos_tags: List[Tuple]) -> float:
        """Analyze phrase complexity"""
        if not pos_tags:
            return 0.0
        
        # Count complex phrases (simplified)
        complex_phrases = 0
        total_phrases = 0
        
        # Look for prepositional phrases, relative clauses, etc.
        for i, (word, pos) in enumerate(pos_tags):
            if pos == 'IN':  # Preposition
                complex_phrases += 1
            if pos.startswith('DT'):  # Determiner
                total_phrases += 1
        
        return complex_phrases / total_phrases if total_phrases > 0 else 0.0
