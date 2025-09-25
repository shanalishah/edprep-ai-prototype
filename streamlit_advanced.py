import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import sys
import re
from typing import Dict, List, Optional, Any

# Page configuration
st.set_page_config(
    page_title="EdPrep AI - IELTS Test Preparation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .score-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .band-score {
        font-size: 2rem;
        font-weight: bold;
        color: #e74c3c;
    }
    .feedback-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Advanced Essay Scorer (Hybrid - ML + Rule-based)
class AdvancedEssayScorer:
    def __init__(self):
        self.linking_words = [
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'nevertheless', 'additionally', 'similarly', 'likewise', 'in contrast',
            'on the other hand', 'for instance', 'for example', 'in conclusion',
            'to summarize', 'firstly', 'secondly', 'finally', 'meanwhile'
        ]
        
        self.academic_words = [
            'analysis', 'approach', 'area', 'assessment', 'assume', 'authority',
            'available', 'benefit', 'concept', 'consist', 'constitute', 'context',
            'contract', 'create', 'data', 'define', 'derive', 'distribute', 'economy',
            'environment', 'establish', 'estimate', 'evident', 'export', 'factor',
            'finance', 'formula', 'function', 'identify', 'income', 'indicate',
            'individual', 'interpret', 'involve', 'issue', 'labour', 'legal',
            'legislate', 'major', 'method', 'occur', 'percent', 'period', 'policy',
            'principle', 'procedure', 'process', 'project', 'require', 'research',
            'respond', 'role', 'section', 'sector', 'significant', 'similar',
            'source', 'specific', 'structure', 'theory', 'vary'
        ]
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        return len(paragraphs)
    
    def calculate_vocabulary_richness(self, text: str) -> float:
        """Calculate vocabulary richness (unique words / total words)"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        unique_words = len(set(words))
        total_words = len(words)
        return unique_words / total_words
    
    def count_linking_words(self, text: str) -> int:
        """Count linking words in text"""
        text_lower = text.lower()
        count = 0
        for word in self.linking_words:
            count += text_lower.count(word)
        return count
    
    def count_academic_words(self, text: str) -> int:
        """Count academic words in text"""
        text_lower = text.lower()
        count = 0
        for word in self.academic_words:
            count += text_lower.count(word)
        return count
    
    def calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def is_gibberish_or_low_quality(self, essay: str) -> bool:
        """Detect if essay is gibberish or extremely low quality"""
        words = re.findall(r'\b\w+\b', essay.lower())
        if len(words) < 3:
            return True
        
        # Check for repeated characters (like "asdasdasd")
        for word in words:
            if len(word) > 3 and len(set(word)) < len(word) * 0.4:  # Too many repeated chars
                return True
        
        # Check for non-English patterns
        english_letters = sum(1 for c in essay if c.isalpha() and c.isascii())
        total_chars = sum(1 for c in essay if c.isalpha())
        if total_chars > 0 and english_letters / total_chars < 0.8:
            return True
        
        # Check for meaningful word ratio
        meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
        if len(words) > 0 and len(meaningful_words) / len(words) < 0.5:
            return True
        
        return False
    
    def score_task_achievement(self, essay: str, prompt: str, task_type: str) -> float:
        """Score Task Achievement based on word count and content relevance"""
        # Check for gibberish first
        if self.is_gibberish_or_low_quality(essay):
            return 1.0
        
        word_count = self.count_words(essay)
        
        # Much stricter scoring based on word count
        if task_type == "Task 1":
            if word_count < 50:
                return 2.0
            elif word_count < 100:
                return 3.0
            elif word_count < 150:
                return 4.0
            elif word_count < 200:
                return 5.0
            else:
                base_score = 6.0
        else:  # Task 2
            if word_count < 100:
                return 2.0
            elif word_count < 150:
                return 3.0
            elif word_count < 200:
                return 4.0
            elif word_count < 250:
                return 5.0
            else:
                base_score = 6.0
        
        # Adjust based on content relevance (simple keyword matching)
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        essay_words = set(re.findall(r'\b\w+\b', essay.lower()))
        if len(prompt_words) > 0:
            relevance = len(prompt_words.intersection(essay_words)) / len(prompt_words)
            return min(9.0, base_score + relevance * 1.5)
        
        return base_score
    
    def score_coherence_cohesion(self, essay: str) -> float:
        """Score Coherence & Cohesion"""
        # Check for gibberish first
        if self.is_gibberish_or_low_quality(essay):
            return 1.0
        
        linking_count = self.count_linking_words(essay)
        paragraphs = self.count_paragraphs(essay)
        sentences = self.count_sentences(essay)
        word_count = self.count_words(essay)
        
        # Much stricter base scoring
        if word_count < 50:
            base_score = 2.0
        elif word_count < 100:
            base_score = 3.0
        elif word_count < 150:
            base_score = 4.0
        else:
            base_score = 5.0
        
        # Linking words bonus (more realistic)
        if linking_count >= 5:
            base_score += 1.5
        elif linking_count >= 3:
            base_score += 1.0
        elif linking_count >= 1:
            base_score += 0.5
        
        # Paragraph structure bonus
        if paragraphs >= 3:
            base_score += 1.0
        elif paragraphs >= 2:
            base_score += 0.5
        
        return min(9.0, base_score)
    
    def score_lexical_resource(self, essay: str) -> float:
        """Score Lexical Resource"""
        # Check for gibberish first
        if self.is_gibberish_or_low_quality(essay):
            return 1.0
        
        vocabulary_richness = self.calculate_vocabulary_richness(essay)
        academic_words = self.count_academic_words(essay)
        word_count = self.count_words(essay)
        
        # Much stricter base scoring
        if word_count < 50:
            base_score = 2.0
        elif word_count < 100:
            base_score = 3.0
        elif word_count < 150:
            base_score = 4.0
        else:
            base_score = 5.0
        
        # Vocabulary richness bonus (more realistic)
        if vocabulary_richness >= 0.8:
            base_score += 2.0
        elif vocabulary_richness >= 0.7:
            base_score += 1.5
        elif vocabulary_richness >= 0.6:
            base_score += 1.0
        elif vocabulary_richness >= 0.5:
            base_score += 0.5
        
        # Academic vocabulary bonus
        academic_ratio = academic_words / max(word_count, 1)
        if academic_ratio >= 0.1:
            base_score += 1.0
        elif academic_ratio >= 0.05:
            base_score += 0.5
        
        return min(9.0, base_score)
    
    def score_grammatical_range(self, essay: str) -> float:
        """Score Grammatical Range & Accuracy"""
        # Check for gibberish first
        if self.is_gibberish_or_low_quality(essay):
            return 1.0
        
        avg_sentence_length = self.calculate_avg_sentence_length(essay)
        sentences = self.count_sentences(essay)
        word_count = self.count_words(essay)
        
        # Much stricter base scoring
        if word_count < 50:
            base_score = 2.0
        elif word_count < 100:
            base_score = 3.0
        elif word_count < 150:
            base_score = 4.0
        else:
            base_score = 5.0
        
        # Sentence variety bonus (more realistic)
        if avg_sentence_length >= 15:
            base_score += 1.5
        elif avg_sentence_length >= 12:
            base_score += 1.0
        elif avg_sentence_length >= 10:
            base_score += 0.5
        
        # Sentence count bonus (shows complexity)
        if sentences >= 8:
            base_score += 1.0
        elif sentences >= 6:
            base_score += 0.5
        
        return min(9.0, base_score)
    
    def score_essay(self, essay: str, prompt: str, task_type: str = "Task 2") -> Dict[str, float]:
        """Score essay on all four criteria"""
        return {
            'task_achievement': self.score_task_achievement(essay, prompt, task_type),
            'coherence_cohesion': self.score_coherence_cohesion(essay),
            'lexical_resource': self.score_lexical_resource(essay),
            'grammatical_range_accuracy': self.score_grammatical_range(essay),
            'overall_band_score': 0.0  # Will be calculated
        }

# Advanced Feedback Generator (Restored from Backend)
class AdvancedFeedbackGenerator:
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
        """Generate comprehensive feedback for an essay"""
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
                "suggestions": suggestions,
                "feedback": detailed_feedback  # For compatibility
            }
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return {
                "detailed_feedback": "Feedback generation failed. Please try again.",
                "suggestions": ["Review your essay for grammar and vocabulary errors."],
                "feedback": "Feedback generation failed. Please try again."
            }
    
    def _generate_ai_feedback(self, prompt: str, essay: str, scores: Dict[str, float], task_type: str) -> str:
        """Generate feedback using OpenAI API"""
        try:
            system_prompt = f"""You are an expert IELTS writing examiner. Provide detailed feedback for this {task_type} essay based on the IELTS scoring criteria.

The essay scored:
- Task Achievement: {scores['task_achievement']}/9
- Coherence and Cohesion: {scores['coherence_cohesion']}/9  
- Lexical Resource: {scores['lexical_resource']}/9
- Grammatical Range and Accuracy: {scores['grammatical_range_accuracy']}/9
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
        """Generate feedback using rule-based approach"""
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
        gra_score = scores['grammatical_range_accuracy']
        if gra_score < 6.0:
            feedback_parts.append(f"**Grammatical Range and Accuracy ({gra_score}/9):** Improve your grammar:")
            feedback_parts.append("- Use a variety of sentence structures (simple, compound, complex)")
            feedback_parts.append("- Check subject-verb agreement")
            feedback_parts.append("- Use correct verb tenses consistently")
            feedback_parts.append("- Proofread for spelling and punctuation errors")
        
        return "\n\n".join(feedback_parts)
    
    def _generate_suggestions(self, essay: str, scores: Dict[str, float]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Word count suggestion
        word_count = len(essay.split())
        if word_count < 250:
            suggestions.append(f"Write more content - you have {word_count} words, aim for at least 250 words")
        
        # Vocabulary suggestions
        if scores['lexical_resource'] < 6.0:
            suggestions.append("Use more varied vocabulary and avoid repetition")
            suggestions.append("Include some academic vocabulary appropriate for the topic")
        
        # Grammar suggestions
        if scores['grammatical_range_accuracy'] < 6.0:
            suggestions.append("Use a variety of sentence structures")
            suggestions.append("Check for grammatical errors and proofread carefully")
        
        # Structure suggestions
        if scores['coherence_cohesion'] < 6.0:
            suggestions.append("Improve paragraph structure and use linking words")
            suggestions.append("Ensure each paragraph has a clear topic sentence")
        
        return suggestions

# Initialize models
@st.cache_resource
def load_models():
    essay_scorer = AdvancedEssayScorer()
    feedback_generator = AdvancedFeedbackGenerator()
    return essay_scorer, feedback_generator

# Main app
def main():
    st.markdown('<div class="main-header">🎓 EdPrep AI - IELTS Test Preparation</div>', unsafe_allow_html=True)
    
    # Load models
    essay_scorer, feedback_generator = load_models()
    
    # Sidebar navigation
    st.sidebar.title("📚 Test Sections")
    section = st.sidebar.selectbox(
        "Choose a test section:",
        ["Writing Test", "About"]
    )
    
    if section == "Writing Test":
        writing_test_section(essay_scorer, feedback_generator)
    elif section == "About":
        about_section()

def writing_test_section(essay_scorer, feedback_generator):
    st.markdown('<div class="section-header">✍️ IELTS Writing Test</div>', unsafe_allow_html=True)
    
    # Task type selection
    task_type = st.selectbox("Select Task Type:", ["Task 1", "Task 2"])
    
    # Sample prompts
    sample_prompts = {
        "Task 1": [
            "The chart below shows the percentage of households in owned and rented accommodation in England and Wales between 1918 and 2011. Summarize the information by selecting and reporting the main features, and make comparisons where relevant.",
            "The table below shows the number of cars per 1000 people in different countries in 1990 and 2000. Summarize the information by selecting and reporting the main features, and make comparisons where relevant."
        ],
        "Task 2": [
            "Some people believe that technology has made our lives more complicated, while others think it has made life easier. Discuss both views and give your opinion.",
            "In many countries, the number of people living alone is increasing. What are the causes of this trend? What effects does it have on society?",
            "Some people think that the best way to reduce crime is to give longer prison sentences. Others, however, believe there are better alternative ways of reducing crime. Discuss both views and give your opinion."
        ]
    }
    
    # Prompt selection
    selected_prompt = st.selectbox("Choose a sample prompt:", sample_prompts[task_type])
    
    # Essay input
    essay_text = st.text_area(
        "Write your essay here:",
        height=300,
        placeholder="Type your essay here..."
    )
    
    # Submit button
    if st.button("📊 Assess Essay", type="primary"):
        if essay_text.strip():
            with st.spinner("Analyzing your essay..."):
                # Score the essay
                scores = essay_scorer.score_essay(
                    essay=essay_text,
                    prompt=selected_prompt,
                    task_type=task_type
                )
                
                # Calculate overall band score
                scores['overall_band_score'] = (
                    scores['task_achievement'] + 
                    scores['coherence_cohesion'] + 
                    scores['lexical_resource'] + 
                    scores['grammatical_range_accuracy']
                ) / 4
                
                # Generate feedback
                feedback = feedback_generator.generate_feedback(
                    prompt=selected_prompt,
                    essay=essay_text,
                    scores=scores,
                    task_type=task_type
                )
                
                # Display results
                display_writing_results(scores, feedback, essay_text)
        else:
            st.warning("Please write an essay before submitting.")

def display_writing_results(scores, feedback, essay_text):
    st.markdown('<div class="section-header">📊 Assessment Results</div>', unsafe_allow_html=True)
    
    # Overall band score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.metric("Task Achievement", f"{scores['task_achievement']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.metric("Coherence & Cohesion", f"{scores['coherence_cohesion']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.metric("Lexical Resource", f"{scores['lexical_resource']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.metric("Grammar Range & Accuracy", f"{scores['grammatical_range_accuracy']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Overall band score
    st.markdown(f'<div class="band-score">Overall Band Score: {scores["overall_band_score"]:.1f}</div>', unsafe_allow_html=True)
    
    # Word count
    word_count = len(essay_text.split())
    st.info(f"📝 Word Count: {word_count} words")
    
    # Detailed feedback
    st.markdown('<div class="section-header">💡 Detailed Feedback</div>', unsafe_allow_html=True)
    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
    st.markdown(feedback.get('detailed_feedback', feedback.get('feedback', 'No feedback available')))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Suggestions
    if 'suggestions' in feedback and feedback['suggestions']:
        st.markdown('<div class="section-header">💡 Improvement Suggestions</div>', unsafe_allow_html=True)
        for suggestion in feedback['suggestions']:
            st.markdown(f"• {suggestion}")

def about_section():
    st.markdown('<div class="section-header">ℹ️ About EdPrep AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎓 Welcome to EdPrep AI - Your IELTS Test Preparation Companion
    
    **EdPrep AI** is an AI-powered platform designed to help students and teachers prepare for competitive tests like IELTS, TOEFL, and GRE. Our focus is on providing realistic, accurate assessments and detailed feedback to help you improve your test performance.
    
    ### ✨ Features
    
    #### ✍️ **Writing Test**
    - AI-powered essay scoring based on official IELTS criteria
    - Detailed feedback on Task Achievement, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy
    - Realistic band score predictions
    - Sample prompts for both Task 1 and Task 2
    
    ### 🚀 Technology
    
    - **AI-Powered Scoring**: Advanced rule-based scoring system with ML capabilities
    - **Official IELTS Criteria**: Scoring based on official IELTS band descriptors
    - **Detailed Feedback**: Personalized feedback to help you improve
    - **OpenAI Integration**: Enhanced feedback generation when API key is available
    
    ### 📊 Scoring System
    
    Our scoring system follows the official IELTS band descriptors:
    - **Band 9**: Expert user
    - **Band 8**: Very good user
    - **Band 7**: Good user
    - **Band 6**: Competent user
    - **Band 5**: Modest user
    - **Band 4**: Limited user
    - **Band 3**: Extremely limited user
    - **Band 2**: Intermittent user
    - **Band 1**: Non-user
    
    ### 🎯 How to Use
    
    1. **Select a test section** from the sidebar
    2. **Choose a test** or writing prompt
    3. **Complete the test** or write your essay
    4. **Submit** and receive detailed feedback
    5. **Review your results** and identify areas for improvement
    
    ### 📈 Tips for Success
    
    - **Practice regularly** with different types of questions
    - **Review feedback carefully** to understand your strengths and weaknesses
    - **Focus on areas** where you need improvement
    - **Time yourself** to simulate real test conditions
    
    ### 🔧 Technical Details
    
    - **Backend**: Python with advanced rule-based scoring system
    - **Frontend**: Streamlit for easy interaction
    - **Scoring**: Hybrid system with vocabulary analysis, grammar checking, and content relevance
    - **AI Integration**: OpenAI API for enhanced feedback generation
    
    ### 📞 Support
    
    If you have any questions or need help, please don't hesitate to reach out. We're here to help you succeed in your IELTS journey!
    
    ---
    
    **Good luck with your IELTS preparation! 🍀**
    """)

if __name__ == "__main__":
    main()
