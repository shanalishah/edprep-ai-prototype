import re
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

load_dotenv()

class FeedbackGenerator:
    """
    Generate detailed feedback for IELTS essays
    """
    
    def __init__(self):
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        try:
            if os.getenv("OPENAI_API_KEY"):
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.openai_client = openai
        except ImportError:
            print("OpenAI not available - using rule-based feedback only")
    
    def generate_feedback(self, prompt: str, essay: str, scores: Dict[str, float], task_type: str = "Task 2") -> Dict[str, Any]:
        """
        Generate comprehensive feedback for an essay
        """
        try:
            # Generate detailed feedback using AI if available, otherwise use rule-based
            if self.openai_client:
                detailed_feedback = self._generate_ai_feedback(prompt, essay, scores, task_type)
            else:
                detailed_feedback = self._generate_rule_based_feedback(prompt, essay, scores, task_type)
            
            # Generate specific suggestions
            suggestions = self._generate_suggestions(essay, scores)
            
            return {
                "detailed_feedback": detailed_feedback,
                "suggestions": suggestions
            }
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return {
                "detailed_feedback": "Feedback generation failed. Please try again.",
                "suggestions": ["Review your essay for grammar and vocabulary errors."]
            }
    
    def _generate_ai_feedback(self, prompt: str, essay: str, scores: Dict[str, float], task_type: str) -> str:
        """
        Generate feedback using OpenAI API
        """
        try:
            system_prompt = f"""You are an expert IELTS writing examiner. Provide detailed feedback for this {task_type} essay based on the IELTS scoring criteria.

The essay scored:
- Task Achievement: {scores['task_achievement']}/9
- Coherence and Cohesion: {scores['coherence_cohesion']}/9  
- Lexical Resource: {scores['lexical_resource']}/9
- Grammatical Range and Accuracy: {scores['grammatical_range']}/9
- Overall Band Score: {scores['overall_band_score']}/9

Provide constructive feedback focusing on:
1. What the student did well
2. Specific areas for improvement
3. Actionable advice for each scoring criterion
4. Examples of how to improve

Keep the feedback encouraging but honest, and provide specific examples from the essay."""

            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Prompt: {prompt}\n\nEssay: {essay}"}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_rule_based_feedback(prompt, essay, scores, task_type)
    
    def _generate_rule_based_feedback(self, prompt: str, essay: str, scores: Dict[str, float], task_type: str) -> str:
        """
        Generate feedback using rule-based approach
        """
        feedback_parts = []
        
        # Overall assessment
        overall_score = scores['overall_band_score']
        if overall_score >= 7.0:
            feedback_parts.append("**Overall Assessment:** This is a strong essay that demonstrates good English proficiency.")
        elif overall_score >= 6.0:
            feedback_parts.append("**Overall Assessment:** This is a good essay with room for improvement in several areas.")
        elif overall_score >= 5.0:
            feedback_parts.append("**Overall Assessment:** This essay shows basic competence but needs significant improvement.")
        else:
            feedback_parts.append("**Overall Assessment:** This essay needs substantial improvement to meet IELTS standards.")
        
        # Task Achievement feedback
        ta_score = scores['task_achievement']
        if ta_score < 6.0:
            feedback_parts.append(f"**Task Achievement ({ta_score}/9):** The essay doesn't fully address the task requirements. Make sure to:")
            feedback_parts.append("- Clearly state your position in the introduction")
            feedback_parts.append("- Develop your arguments with specific examples")
            feedback_parts.append("- Write at least 250 words for Task 2")
            if task_type == "Task 2":
                feedback_parts.append("- Include a clear conclusion that summarizes your main points")
        
        # Coherence and Cohesion feedback
        cc_score = scores['coherence_cohesion']
        if cc_score < 6.0:
            feedback_parts.append(f"**Coherence and Cohesion ({cc_score}/9):** Improve the organization and flow of your essay:")
            feedback_parts.append("- Use clear paragraph structure (introduction, body paragraphs, conclusion)")
            feedback_parts.append("- Add linking words like 'however', 'therefore', 'moreover', 'furthermore'")
            feedback_parts.append("- Start each paragraph with a clear topic sentence")
        
        # Lexical Resource feedback
        lr_score = scores['lexical_resource']
        if lr_score < 6.0:
            feedback_parts.append(f"**Lexical Resource ({lr_score}/9):** Expand your vocabulary:")
            feedback_parts.append("- Use more varied and precise vocabulary")
            feedback_parts.append("- Avoid repeating the same words")
            feedback_parts.append("- Include some academic vocabulary appropriate for the topic")
            feedback_parts.append("- Check word choice for accuracy")
        
        # Grammatical Range and Accuracy feedback
        gra_score = scores['grammatical_range']
        if gra_score < 6.0:
            feedback_parts.append(f"**Grammatical Range and Accuracy ({gra_score}/9):** Improve your grammar:")
            feedback_parts.append("- Use a variety of sentence structures (simple, compound, complex)")
            feedback_parts.append("- Check subject-verb agreement")
            feedback_parts.append("- Use correct verb tenses consistently")
            feedback_parts.append("- Proofread for spelling and punctuation errors")
        
        return "\n\n".join(feedback_parts)
    
    def _generate_suggestions(self, essay: str, scores: Dict[str, float]) -> List[str]:
        """
        Generate specific improvement suggestions
        """
        suggestions = []
        
        # Word count suggestion
        word_count = len(essay.split())
        if word_count < 250:
            suggestions.append(f"Write more! Your essay has {word_count} words. Aim for at least 250 words for Task 2.")
        elif word_count > 350:
            suggestions.append(f"Your essay is quite long ({word_count} words). Consider being more concise.")
        
        # Grammar suggestions
        if scores['grammatical_range'] < 6.0:
            suggestions.append("Practice using complex sentences with conjunctions like 'because', 'although', 'while'.")
            suggestions.append("Review common grammar rules, especially subject-verb agreement and verb tenses.")
        
        # Vocabulary suggestions
        if scores['lexical_resource'] < 6.0:
            suggestions.append("Learn 5-10 new academic words each day and practice using them in sentences.")
            suggestions.append("Use a thesaurus to find synonyms for common words you use frequently.")
        
        # Organization suggestions
        if scores['coherence_cohesion'] < 6.0:
            suggestions.append("Practice writing essays with clear paragraph structure: introduction, 2-3 body paragraphs, conclusion.")
            suggestions.append("Learn and practice using linking words and phrases to connect ideas.")
        
        # Task achievement suggestions
        if scores['task_achievement'] < 6.0:
            suggestions.append("Make sure you fully understand the question before writing.")
            suggestions.append("Plan your essay structure before you start writing.")
            suggestions.append("Include specific examples to support your arguments.")
        
        return suggestions[:5]  # Limit to 5 suggestions
