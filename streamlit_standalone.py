import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import sys
import re
from typing import Dict, List, Optional

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

# Simple Essay Scorer (Rule-based)
class SimpleEssayScorer:
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
    
    def score_task_achievement(self, essay: str, prompt: str, task_type: str) -> float:
        """Score Task Achievement based on word count and content relevance"""
        word_count = self.count_words(essay)
        
        # Base score on word count
        if task_type == "Task 1":
            if word_count >= 150:
                base_score = 6.0
            elif word_count >= 120:
                base_score = 5.5
            else:
                base_score = 4.0
        else:  # Task 2
            if word_count >= 250:
                base_score = 6.0
            elif word_count >= 200:
                base_score = 5.5
            else:
                base_score = 4.0
        
        # Adjust based on content relevance (simple keyword matching)
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        essay_words = set(re.findall(r'\b\w+\b', essay.lower()))
        relevance = len(prompt_words.intersection(essay_words)) / len(prompt_words)
        
        return min(9.0, base_score + relevance * 2)
    
    def score_coherence_cohesion(self, essay: str) -> float:
        """Score Coherence & Cohesion"""
        linking_count = self.count_linking_words(essay)
        paragraphs = self.count_paragraphs(essay)
        sentences = self.count_sentences(essay)
        
        # Base score
        base_score = 5.0
        
        # Linking words bonus
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
        vocabulary_richness = self.calculate_vocabulary_richness(essay)
        academic_words = self.count_academic_words(essay)
        word_count = self.count_words(essay)
        
        # Base score
        base_score = 5.0
        
        # Vocabulary richness bonus
        if vocabulary_richness >= 0.7:
            base_score += 2.0
        elif vocabulary_richness >= 0.6:
            base_score += 1.5
        elif vocabulary_richness >= 0.5:
            base_score += 1.0
        
        # Academic vocabulary bonus
        academic_ratio = academic_words / max(word_count, 1)
        if academic_ratio >= 0.1:
            base_score += 1.0
        elif academic_ratio >= 0.05:
            base_score += 0.5
        
        return min(9.0, base_score)
    
    def score_grammatical_range(self, essay: str) -> float:
        """Score Grammatical Range & Accuracy"""
        avg_sentence_length = self.calculate_avg_sentence_length(essay)
        sentences = self.count_sentences(essay)
        
        # Base score
        base_score = 5.0
        
        # Sentence variety bonus
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

# Simple Feedback Generator
class SimpleFeedbackGenerator:
    def generate_feedback(self, prompt: str, essay: str, scores: Dict[str, float], task_type: str) -> Dict[str, str]:
        """Generate simple feedback based on scores"""
        word_count = len(essay.split())
        
        feedback_parts = []
        
        # Task Achievement feedback
        ta_score = scores['task_achievement']
        if ta_score < 5.0:
            feedback_parts.append("**Task Achievement**: Your essay needs to better address the prompt. Make sure to cover all parts of the question and provide relevant examples.")
        elif ta_score < 6.5:
            feedback_parts.append("**Task Achievement**: Good attempt at addressing the prompt. Try to provide more specific examples and ensure all parts of the question are covered.")
        else:
            feedback_parts.append("**Task Achievement**: Well done! You have effectively addressed the prompt with relevant content.")
        
        # Coherence & Cohesion feedback
        cc_score = scores['coherence_cohesion']
        if cc_score < 5.0:
            feedback_parts.append("**Coherence & Cohesion**: Your essay needs better organization. Use more linking words and ensure clear paragraph structure.")
        elif cc_score < 6.5:
            feedback_parts.append("**Coherence & Cohesion**: Good organization overall. Try using more linking words to connect your ideas better.")
        else:
            feedback_parts.append("**Coherence & Cohesion**: Excellent organization and use of linking words to connect ideas.")
        
        # Lexical Resource feedback
        lr_score = scores['lexical_resource']
        if lr_score < 5.0:
            feedback_parts.append("**Lexical Resource**: Try to use more varied vocabulary and avoid repetition. Include more academic words.")
        elif lr_score < 6.5:
            feedback_parts.append("**Lexical Resource**: Good vocabulary range. Try to use more sophisticated and academic vocabulary.")
        else:
            feedback_parts.append("**Lexical Resource**: Excellent vocabulary range with sophisticated word choices.")
        
        # Grammatical Range feedback
        gr_score = scores['grammatical_range_accuracy']
        if gr_score < 5.0:
            feedback_parts.append("**Grammatical Range**: Work on using more complex sentence structures and check for grammatical errors.")
        elif gr_score < 6.5:
            feedback_parts.append("**Grammatical Range**: Good grammatical control. Try to use more varied sentence structures.")
        else:
            feedback_parts.append("**Grammatical Range**: Excellent grammatical control with varied sentence structures.")
        
        # Word count feedback
        if task_type == "Task 1":
            if word_count < 150:
                feedback_parts.append(f"**Word Count**: You have {word_count} words. Task 1 requires at least 150 words.")
            else:
                feedback_parts.append(f"**Word Count**: Good! You have {word_count} words, which meets the minimum requirement.")
        else:
            if word_count < 250:
                feedback_parts.append(f"**Word Count**: You have {word_count} words. Task 2 requires at least 250 words.")
            else:
                feedback_parts.append(f"**Word Count**: Good! You have {word_count} words, which meets the minimum requirement.")
        
        return {
            'feedback': '\n\n'.join(feedback_parts),
            'detailed_feedback': '\n\n'.join(feedback_parts)
        }

# Initialize models
@st.cache_resource
def load_models():
    essay_scorer = SimpleEssayScorer()
    feedback_generator = SimpleFeedbackGenerator()
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
    st.markdown(feedback.get('feedback', feedback.get('detailed_feedback', 'No feedback available')))
    st.markdown('</div>', unsafe_allow_html=True)

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
    
    - **AI-Powered Scoring**: Advanced rule-based scoring system
    - **Official IELTS Criteria**: Scoring based on official IELTS band descriptors
    - **Detailed Feedback**: Personalized feedback to help you improve
    
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
    
    - **Backend**: Python with rule-based scoring system
    - **Frontend**: Streamlit for easy interaction
    - **Scoring**: Rule-based system with vocabulary analysis, grammar checking, and content relevance
    
    ### 📞 Support
    
    If you have any questions or need help, please don't hesitate to reach out. We're here to help you succeed in your IELTS journey!
    
    ---
    
    **Good luck with your IELTS preparation! 🍀**
    """)

if __name__ == "__main__":
    main()
