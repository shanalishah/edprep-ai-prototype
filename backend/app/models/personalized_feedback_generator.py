"""
Personalized Feedback Generator
Uses advanced semantic analysis to provide detailed, personalized feedback
"""

from typing import Dict, List, Any, Optional
from .advanced_semantic_analyzer import AdvancedSemanticAnalyzer

class PersonalizedFeedbackGenerator:
    """
    Generates personalized, detailed feedback based on advanced semantic analysis
    """
    
    def __init__(self):
        self.semantic_analyzer = AdvancedSemanticAnalyzer()
        
        # Feedback templates for different score ranges
        self.feedback_templates = {
            'task_achievement': {
                'high': "Excellent task achievement! You have fully addressed the prompt with clear position and well-developed arguments.",
                'medium': "Good task achievement, but there's room for improvement in addressing all parts of the question.",
                'low': "Task achievement needs significant improvement. Focus on fully addressing the prompt requirements."
            },
            'coherence_cohesion': {
                'high': "Outstanding coherence and cohesion! Your essay flows logically with excellent use of linking devices.",
                'medium': "Good organization, but work on improving the logical flow and use of cohesive devices.",
                'low': "Coherence and cohesion need major improvement. Focus on clear paragraph structure and linking words."
            },
            'lexical_resource': {
                'high': "Excellent vocabulary range and accuracy! You demonstrate sophisticated lexical control.",
                'medium': "Good vocabulary use, but try to expand your range and improve word choice accuracy.",
                'low': "Lexical resource needs significant improvement. Focus on expanding vocabulary and improving accuracy."
            },
            'grammatical_range': {
                'high': "Excellent grammatical range and accuracy! You demonstrate sophisticated sentence structures.",
                'medium': "Good grammar overall, but work on expanding sentence variety and reducing errors.",
                'low': "Grammatical range and accuracy need major improvement. Focus on basic grammar and sentence variety."
            }
        }
    
    def generate_personalized_feedback(self, essay: str, prompt: str, task_type: str, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive, personalized feedback
        """
        # Perform advanced semantic analysis
        semantic_analysis = self.semantic_analyzer.analyze_essay_semantics(essay, prompt, task_type)
        
        # Generate personalized feedback for each criterion
        feedback = {
            'overall_assessment': self._generate_overall_assessment(scores, semantic_analysis),
            'criterion_feedback': self._generate_criterion_feedback(scores, semantic_analysis),
            'strengths': self._identify_strengths(semantic_analysis),
            'weaknesses': self._identify_weaknesses(semantic_analysis),
            'specific_suggestions': self._generate_specific_suggestions(semantic_analysis, scores),
            'improvement_plan': self._create_improvement_plan(semantic_analysis, scores),
            'semantic_insights': self._generate_semantic_insights(semantic_analysis)
        }
        
        return feedback
    
    def _generate_overall_assessment(self, scores: Dict[str, float], analysis: Dict[str, Any]) -> str:
        """Generate overall assessment"""
        overall_score = scores['overall_band_score']
        
        if overall_score >= 7.0:
            return f"**Outstanding Performance (Band {overall_score})**\n\nYour essay demonstrates excellent English proficiency with sophisticated language use and clear argumentation. You're well-prepared for academic or professional contexts."
        elif overall_score >= 6.0:
            return f"**Good Performance (Band {overall_score})**\n\nYour essay shows good English proficiency with some areas for improvement. With focused practice, you can achieve higher scores."
        elif overall_score >= 5.0:
            return f"**Moderate Performance (Band {overall_score})**\n\nYour essay demonstrates basic competence but needs significant improvement in several areas. Focus on the specific weaknesses identified below."
        else:
            return f"**Needs Improvement (Band {overall_score})**\n\nYour essay requires substantial improvement across all criteria. Focus on building fundamental skills before attempting more advanced techniques."
    
    def _generate_criterion_feedback(self, scores: Dict[str, float], analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate detailed feedback for each criterion"""
        criterion_feedback = {}
        
        for criterion in ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range']:
            score = scores[criterion]
            feedback = self._get_criterion_feedback(criterion, score, analysis)
            criterion_feedback[criterion] = feedback
        
        return criterion_feedback
    
    def _get_criterion_feedback(self, criterion: str, score: float, analysis: Dict[str, Any]) -> str:
        """Get specific feedback for a criterion"""
        if score >= 7.0:
            level = 'high'
        elif score >= 5.0:
            level = 'medium'
        else:
            level = 'low'
        
        base_feedback = self.feedback_templates[criterion][level]
        
        # Add specific insights based on analysis
        specific_insights = self._get_specific_insights(criterion, analysis)
        
        return f"{base_feedback}\n\n{specific_insights}"
    
    def _get_specific_insights(self, criterion: str, analysis: Dict[str, Any]) -> str:
        """Get specific insights for a criterion"""
        insights = []
        
        if criterion == 'task_achievement':
            task_relevance = analysis['task_relevance']
            if task_relevance['relevance_score'] < 0.5:
                insights.append("• **Task Relevance**: Your essay doesn't fully address the prompt. Make sure to cover all parts of the question.")
            if task_relevance['position_clarity'] < 0.5:
                insights.append("• **Position Clarity**: Your position isn't clear enough. State your opinion explicitly in the introduction.")
            if task_relevance['argument_development'] < 0.5:
                insights.append("• **Argument Development**: Your arguments need more development. Provide specific examples and explanations.")
        
        elif criterion == 'coherence_cohesion':
            discourse_markers = analysis['discourse_markers']
            if discourse_markers['variety_score'] < 0.3:
                insights.append("• **Discourse Markers**: Use more variety in linking words. Try 'furthermore', 'however', 'consequently'.")
            if discourse_markers['marker_density'] < 0.2:
                insights.append("• **Cohesive Devices**: Add more linking words to connect your ideas smoothly.")
            
            basic_metrics = analysis['basic_metrics']
            if basic_metrics['paragraph_count'] < 3:
                insights.append("• **Paragraph Structure**: Use clear paragraph structure: introduction, body paragraphs, conclusion.")
        
        elif criterion == 'lexical_resource':
            vocab_soph = analysis['vocabulary_sophistication']
            if vocab_soph['lexical_diversity'] < 0.5:
                insights.append("• **Lexical Diversity**: Avoid repeating the same words. Use synonyms and varied vocabulary.")
            if vocab_soph['advanced_vocabulary_ratio'] < 0.1:
                insights.append("• **Vocabulary Level**: Use more sophisticated vocabulary. Try words like 'significant', 'consequently', 'furthermore'.")
            if vocab_soph['academic_vocabulary_ratio'] < 0.05:
                insights.append("• **Academic Language**: Include more academic vocabulary appropriate for IELTS.")
        
        elif criterion == 'grammatical_range':
            grammar_analysis = analysis['grammatical_complexity']
            if grammar_analysis['sentence_variety_score'] < 0.5:
                insights.append("• **Sentence Variety**: Mix simple, compound, and complex sentences for better variety.")
            if grammar_analysis['grammar_errors'] > 0:
                insights.append("• **Grammar Accuracy**: Check for subject-verb agreement and other common errors.")
            if grammar_analysis['passive_voice_ratio'] < 0.1:
                insights.append("• **Grammar Range**: Try using passive voice occasionally for variety.")
        
        return '\n'.join(insights) if insights else "• Continue practicing to improve in this area."
    
    def _identify_strengths(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify essay strengths"""
        strengths = []
        
        # Vocabulary strengths
        vocab_soph = analysis['vocabulary_sophistication']
        if vocab_soph['lexical_diversity'] > 0.6:
            strengths.append("Good lexical diversity - you use varied vocabulary")
        if vocab_soph['advanced_vocabulary_ratio'] > 0.15:
            strengths.append("Sophisticated vocabulary use")
        if vocab_soph['academic_vocabulary_ratio'] > 0.1:
            strengths.append("Good use of academic vocabulary")
        
        # Grammar strengths
        grammar_analysis = analysis['grammatical_complexity']
        if grammar_analysis['sentence_variety_score'] > 0.7:
            strengths.append("Excellent sentence variety")
        if grammar_analysis['grammar_errors'] == 0:
            strengths.append("Good grammar accuracy")
        if grammar_analysis['passive_voice_ratio'] > 0.1:
            strengths.append("Good use of passive voice")
        
        # Coherence strengths
        discourse_markers = analysis['discourse_markers']
        if discourse_markers['variety_score'] > 0.6:
            strengths.append("Good variety in linking words")
        if discourse_markers['marker_density'] > 0.3:
            strengths.append("Effective use of cohesive devices")
        
        # Readability strengths
        readability = analysis['readability_metrics']
        if readability['flesch_reading_ease'] > 60:
            strengths.append("Good readability and clarity")
        
        return strengths if strengths else ["Keep working on improving your writing skills"]
    
    def _identify_weaknesses(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify essay weaknesses"""
        weaknesses = []
        
        # Vocabulary weaknesses
        vocab_soph = analysis['vocabulary_sophistication']
        if vocab_soph['lexical_diversity'] < 0.4:
            weaknesses.append("Limited lexical diversity - avoid repeating words")
        if vocab_soph['repetition_penalty'] > 0:
            weaknesses.append("Word repetition - use synonyms instead")
        if vocab_soph['advanced_vocabulary_ratio'] < 0.05:
            weaknesses.append("Limited sophisticated vocabulary")
        
        # Grammar weaknesses
        grammar_analysis = analysis['grammatical_complexity']
        if grammar_analysis['grammar_errors'] > 0:
            weaknesses.append("Grammar errors - check subject-verb agreement")
        if grammar_analysis['sentence_variety_score'] < 0.4:
            weaknesses.append("Limited sentence variety")
        if grammar_analysis['tense_consistency'] < 0.7:
            weaknesses.append("Tense consistency issues")
        
        # Coherence weaknesses
        discourse_markers = analysis['discourse_markers']
        if discourse_markers['variety_score'] < 0.3:
            weaknesses.append("Limited variety in linking words")
        if discourse_markers['marker_density'] < 0.1:
            weaknesses.append("Insufficient cohesive devices")
        
        # Task achievement weaknesses
        task_relevance = analysis['task_relevance']
        if task_relevance['relevance_score'] < 0.5:
            weaknesses.append("Poor task relevance - address the prompt more directly")
        if task_relevance['position_clarity'] < 0.5:
            weaknesses.append("Unclear position - state your opinion explicitly")
        
        return weaknesses if weaknesses else ["Continue practicing to improve your writing skills"]
    
    def _generate_specific_suggestions(self, analysis: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Generate specific, actionable suggestions"""
        suggestions = []
        
        # Word count suggestions
        basic_metrics = analysis['basic_metrics']
        if basic_metrics['word_count'] < 250:
            suggestions.append(f"**Word Count**: Write at least 250 words. You currently have {basic_metrics['word_count']} words.")
        
        # Vocabulary suggestions
        vocab_soph = analysis['vocabulary_sophistication']
        if vocab_soph['lexical_diversity'] < 0.5:
            suggestions.append("**Vocabulary**: Learn 5-10 new words daily and practice using them in sentences.")
        if vocab_soph['repetition_penalty'] > 0:
            suggestions.append("**Word Repetition**: Use a thesaurus to find synonyms for common words you repeat.")
        
        # Grammar suggestions
        grammar_analysis = analysis['grammatical_complexity']
        if grammar_analysis['sentence_variety_score'] < 0.5:
            suggestions.append("**Sentence Variety**: Practice writing complex sentences with subordinating conjunctions.")
        if grammar_analysis['grammar_errors'] > 0:
            suggestions.append("**Grammar**: Review basic grammar rules, especially subject-verb agreement.")
        
        # Coherence suggestions
        discourse_markers = analysis['discourse_markers']
        if discourse_markers['variety_score'] < 0.4:
            suggestions.append("**Linking Words**: Learn and practice using different types of linking words.")
        
        # Structure suggestions
        if basic_metrics['paragraph_count'] < 3:
            suggestions.append("**Structure**: Use clear paragraph structure: introduction, 2-3 body paragraphs, conclusion.")
        
        # Task-specific suggestions
        task_relevance = analysis['task_relevance']
        if task_relevance['position_clarity'] < 0.5:
            suggestions.append("**Position**: Clearly state your opinion in the introduction and conclusion.")
        if task_relevance['argument_development'] < 0.5:
            suggestions.append("**Arguments**: Develop your arguments with specific examples and explanations.")
        
        return suggestions if suggestions else ["Continue practicing to improve your writing skills"]
    
    def _create_improvement_plan(self, analysis: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
        """Create a personalized improvement plan"""
        plan = {
            'immediate_focus': [],
            'short_term_goals': [],
            'long_term_goals': [],
            'practice_activities': []
        }
        
        # Identify lowest scoring criterion
        lowest_criterion = min(scores.items(), key=lambda x: x[1])
        
        # Immediate focus (next 1-2 weeks)
        if lowest_criterion[0] == 'task_achievement':
            plan['immediate_focus'].append("Focus on fully addressing the prompt requirements")
            plan['immediate_focus'].append("Practice stating clear positions in introductions")
        elif lowest_criterion[0] == 'coherence_cohesion':
            plan['immediate_focus'].append("Work on paragraph structure and linking words")
            plan['immediate_focus'].append("Practice organizing ideas logically")
        elif lowest_criterion[0] == 'lexical_resource':
            plan['immediate_focus'].append("Expand vocabulary and reduce word repetition")
            plan['immediate_focus'].append("Learn academic vocabulary")
        else:  # grammatical_range
            plan['immediate_focus'].append("Focus on grammar accuracy and sentence variety")
            plan['immediate_focus'].append("Practice complex sentence structures")
        
        # Short-term goals (1-2 months)
        plan['short_term_goals'].append(f"Improve {lowest_criterion[0]} from {lowest_criterion[1]} to {lowest_criterion[1] + 1.0}")
        plan['short_term_goals'].append("Write at least 3 practice essays per week")
        plan['short_term_goals'].append("Focus on one weak area per week")
        
        # Long-term goals (3-6 months)
        overall_score = scores['overall_band_score']
        if overall_score < 6.0:
            plan['long_term_goals'].append("Achieve Band 6.0 or higher")
        elif overall_score < 7.0:
            plan['long_term_goals'].append("Achieve Band 7.0 or higher")
        else:
            plan['long_term_goals'].append("Maintain high performance and aim for Band 8.0+")
        
        # Practice activities
        plan['practice_activities'].append("Daily vocabulary building (10 new words)")
        plan['practice_activities'].append("Weekly essay writing practice")
        plan['practice_activities'].append("Grammar exercises focusing on weak areas")
        plan['practice_activities'].append("Reading academic texts to improve vocabulary")
        
        return plan
    
    def _generate_semantic_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from semantic analysis"""
        insights = {
            'writing_style': self._analyze_writing_style(analysis),
            'language_sophistication': self._analyze_language_sophistication(analysis),
            'coherence_quality': self._analyze_coherence_quality(analysis),
            'academic_readiness': self._analyze_academic_readiness(analysis)
        }
        
        return insights
    
    def _analyze_writing_style(self, analysis: Dict[str, Any]) -> str:
        """Analyze writing style"""
        sentiment = analysis['sentiment_analysis']
        academic_lang = analysis['academic_language']
        
        if academic_lang['academic_language_score'] > 0.7:
            return "Your writing style is appropriately academic and formal, suitable for IELTS and academic contexts."
        elif academic_lang['academic_language_score'] > 0.4:
            return "Your writing style is moderately academic but could be more formal for IELTS standards."
        else:
            return "Your writing style is too informal for IELTS. Focus on using more academic language and formal tone."
    
    def _analyze_language_sophistication(self, analysis: Dict[str, Any]) -> str:
        """Analyze language sophistication"""
        vocab_soph = analysis['vocabulary_sophistication']
        grammar_analysis = analysis['grammatical_complexity']
        
        if vocab_soph['advanced_vocabulary_ratio'] > 0.15 and grammar_analysis['sentence_variety_score'] > 0.7:
            return "Your language use is sophisticated with advanced vocabulary and varied sentence structures."
        elif vocab_soph['advanced_vocabulary_ratio'] > 0.05 or grammar_analysis['sentence_variety_score'] > 0.5:
            return "Your language use shows some sophistication but has room for improvement."
        else:
            return "Your language use is basic. Focus on expanding vocabulary and sentence variety."
    
    def _analyze_coherence_quality(self, analysis: Dict[str, Any]) -> str:
        """Analyze coherence quality"""
        discourse_markers = analysis['discourse_markers']
        semantic_coherence = analysis['semantic_coherence']
        
        if discourse_markers['variety_score'] > 0.6 and semantic_coherence['topic_consistency'] > 0.7:
            return "Your essay demonstrates excellent coherence with good use of linking devices and consistent topic focus."
        elif discourse_markers['variety_score'] > 0.3 or semantic_coherence['topic_consistency'] > 0.5:
            return "Your essay shows reasonable coherence but could benefit from better organization and linking."
        else:
            return "Your essay lacks coherence. Focus on clear organization and effective use of linking words."
    
    def _analyze_academic_readiness(self, analysis: Dict[str, Any]) -> str:
        """Analyze academic readiness"""
        academic_lang = analysis['academic_language']
        readability = analysis['readability_metrics']
        
        if academic_lang['academic_language_score'] > 0.6 and readability['flesch_reading_ease'] > 50:
            return "Your writing demonstrates good academic readiness with appropriate language and clarity."
        elif academic_lang['academic_language_score'] > 0.3:
            return "Your writing shows some academic readiness but needs improvement in formality and clarity."
        else:
            return "Your writing needs significant improvement to meet academic standards. Focus on formal language and clear expression."
