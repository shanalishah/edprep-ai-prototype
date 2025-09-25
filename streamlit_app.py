import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import our models with better error handling
try:
    # Try to import dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # If dotenv is not available, continue without it
        pass
    
    from backend.app.models.essay_scorer import EssayScorer
    from backend.app.models.feedback_generator import FeedbackGenerator
    from backend.app.models.listening_test_data import get_listening_test, get_all_listening_tests
    from backend.app.models.listening_scorer import ListeningScorer, UserListeningAnswer, ListeningTestSubmission
    from backend.app.models.reading_test_data import get_reading_test, get_all_reading_tests
    from backend.app.models.reading_scorer import ReadingScorer, UserAnswer, ReadingTestSubmission
    MODELS_LOADED = True
except ImportError as e:
    st.error(f"Error loading models: {e}")
    MODELS_LOADED = False

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

# Initialize models
@st.cache_resource
def load_models():
    if MODELS_LOADED:
        essay_scorer = EssayScorer()
        feedback_generator = FeedbackGenerator()
        listening_scorer = ListeningScorer()
        reading_scorer = ReadingScorer()
        return essay_scorer, feedback_generator, listening_scorer, reading_scorer
    return None, None, None, None

# Main app
def main():
    st.markdown('<div class="main-header">🎓 EdPrep AI - IELTS Test Preparation</div>', unsafe_allow_html=True)
    
    if not MODELS_LOADED:
        st.error("❌ Models could not be loaded. Please check the backend setup.")
        return
    
    # Load models
    essay_scorer, feedback_generator, listening_scorer, reading_scorer = load_models()
    
    # Sidebar navigation
    st.sidebar.title("📚 Test Sections")
    section = st.sidebar.selectbox(
        "Choose a test section:",
        ["Writing Test", "Reading Test", "Listening Test", "About"]
    )
    
    if section == "Writing Test":
        writing_test_section(essay_scorer, feedback_generator)
    elif section == "Reading Test":
        reading_test_section(reading_scorer)
    elif section == "Listening Test":
        listening_test_section(listening_scorer)
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

def reading_test_section(reading_scorer):
    st.markdown('<div class="section-header">📖 IELTS Reading Test</div>', unsafe_allow_html=True)
    
    # Get available tests
    try:
        tests = get_all_reading_tests()
        if not tests:
            st.warning("No reading tests available.")
            return
        
        # Test selection
        test_options = {f"{test.title} (ID: {test.id})": test for test in tests}
        selected_test_name = st.selectbox("Select a reading test:", list(test_options.keys()))
        selected_test = test_options[selected_test_name]
        
        st.info(f"📚 **{selected_test.title}** - {selected_test.total_questions} questions, {selected_test.time_limit} minutes")
        
        # Display passage
        if selected_test.passages:
            st.markdown("### 📄 Reading Passage")
            for i, passage in enumerate(selected_test.passages, 1):
                st.markdown(f"**Passage {i}:**")
                st.markdown(passage.content)
        
        # Questions and answers
        st.markdown("### ❓ Questions")
        user_answers = {}
        
        for question in selected_test.questions:
            st.markdown(f"**Question {question.question_number}:** {question.question_text}")
            
            if question.type.value == "multiple_choice":
                answer = st.radio(
                    f"Choose your answer for Question {question.question_number}:",
                    question.options,
                    key=f"q{question.id}"
                )
                user_answers[question.id] = answer
            else:
                answer = st.text_input(
                    f"Your answer for Question {question.question_number}:",
                    key=f"q{question.id}"
                )
                user_answers[question.id] = answer
        
        # Submit button
        if st.button("📊 Submit Reading Test", type="primary"):
            if user_answers:
                with st.spinner("Grading your test..."):
                    # Convert to UserAnswer objects
                    user_answer_objects = [
                        UserAnswer(question_id=qid, answer=answer, time_taken=0.0)
                        for qid, answer in user_answers.items()
                    ]
                    
                    # Submit test
                    submission = ReadingTestSubmission(
                        test_id=selected_test.id,
                        answers=user_answer_objects
                    )
                    
                    # Score the test
                    result = reading_scorer.score_test(submission, selected_test)
                    
                    # Display results
                    display_reading_results(result)
            else:
                st.warning("Please answer at least one question before submitting.")
    
    except Exception as e:
        st.error(f"Error loading reading tests: {e}")

def display_reading_results(result):
    st.markdown('<div class="section-header">📊 Reading Test Results</div>', unsafe_allow_html=True)
    
    # Overall score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Correct Answers", f"{result.correct_answers}/{result.total_questions}")
    
    with col2:
        st.metric("Band Score", f"{result.band_score:.1f}")
    
    with col3:
        st.metric("Percentage", f"{result.percentage:.1f}%")
    
    # Detailed feedback
    st.markdown("### 💡 Detailed Feedback")
    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
    if isinstance(result.detailed_feedback, str):
        st.markdown(result.detailed_feedback)
    else:
        st.markdown("Feedback available in detailed analysis.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question-by-question analysis
    st.markdown("### 📋 Question-by-Question Analysis")
    for analysis in result.question_analysis:
        with st.expander(f"Question {analysis.question_id}"):
            st.write(f"**Your Answer:** {analysis.user_answer}")
            st.write(f"**Correct Answer:** {analysis.correct_answer}")
            st.write(f"**Result:** {'✅ Correct' if analysis.is_correct else '❌ Incorrect'}")
            if analysis.explanation:
                st.write(f"**Explanation:** {analysis.explanation}")

def listening_test_section(listening_scorer):
    st.markdown('<div class="section-header">🎧 IELTS Listening Test</div>', unsafe_allow_html=True)
    
    # Get available tests
    try:
        tests = get_all_listening_tests()
        if not tests:
            st.warning("No listening tests available.")
            return
        
        # Test selection
        test_options = {f"{test.title} (ID: {test.id})": test for test in tests}
        selected_test_name = st.selectbox("Select a listening test:", list(test_options.keys()))
        selected_test = test_options[selected_test_name]
        
        st.info(f"🎧 **{selected_test.title}** - {selected_test.total_questions} questions, {selected_test.time_limit} minutes")
        
        # Audio player (placeholder)
        st.markdown("### 🎵 Audio Player")
        st.info("🎧 Audio playback would be available in the full version. For now, please refer to the audio files in your Cambridge IELTS folder.")
        
        # Questions and answers
        st.markdown("### ❓ Questions")
        user_answers = {}
        
        for question in selected_test.questions:
            st.markdown(f"**Question {question.question_number}:** {question.question_text}")
            if question.context:
                st.markdown(f"*{question.context}*")
            
            answer = st.text_input(
                f"Your answer for Question {question.question_number}:",
                key=f"lq{question.id}"
            )
            user_answers[question.id] = answer
        
        # Submit button
        if st.button("📊 Submit Listening Test", type="primary"):
            if user_answers:
                with st.spinner("Grading your test..."):
                    # Convert to UserListeningAnswer objects
                    user_answer_objects = [
                        UserListeningAnswer(question_id=qid, answer=answer, time_taken=0.0)
                        for qid, answer in user_answers.items()
                    ]
                    
                    # Submit test
                    submission = ListeningTestSubmission(
                        test_id=selected_test.id,
                        answers=user_answer_objects
                    )
                    
                    # Score the test
                    result = listening_scorer.score_test(submission, selected_test)
                    
                    # Display results
                    display_listening_results(result)
            else:
                st.warning("Please answer at least one question before submitting.")
    
    except Exception as e:
        st.error(f"Error loading listening tests: {e}")

def display_listening_results(result):
    st.markdown('<div class="section-header">📊 Listening Test Results</div>', unsafe_allow_html=True)
    
    # Overall score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Correct Answers", f"{result.correct_answers}/{result.total_questions}")
    
    with col2:
        st.metric("Band Score", f"{result.band_score:.1f}")
    
    with col3:
        st.metric("Percentage", f"{result.percentage:.1f}%")
    
    # Detailed feedback
    st.markdown("### 💡 Detailed Feedback")
    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
    if isinstance(result.detailed_feedback, str):
        st.markdown(result.detailed_feedback)
    else:
        st.markdown("Feedback available in detailed analysis.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question-by-question analysis
    st.markdown("### 📋 Question-by-Question Analysis")
    for analysis in result.question_analysis:
        with st.expander(f"Question {analysis.question_id}"):
            st.write(f"**Your Answer:** {analysis.user_answer}")
            st.write(f"**Correct Answer:** {analysis.correct_answer}")
            st.write(f"**Result:** {'✅ Correct' if analysis.is_correct else '❌ Incorrect'}")
            if analysis.explanation:
                st.write(f"**Explanation:** {analysis.explanation}")

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
    
    #### 📖 **Reading Test**
    - Authentic Cambridge IELTS reading passages
    - Multiple question types (Multiple Choice, True/False/Not Given, etc.)
    - Detailed explanations for each answer
    - Band score conversion based on official IELTS standards
    
    #### 🎧 **Listening Test**
    - Real Cambridge IELTS listening tests
    - Audio playback support
    - Various question types (Form Completion, Table Completion, etc.)
    - Comprehensive feedback and explanations
    
    ### 🚀 Technology
    
    - **AI-Powered Scoring**: Advanced machine learning models trained on thousands of IELTS essays
    - **Official IELTS Criteria**: Scoring based on official IELTS band descriptors
    - **Real Test Content**: Authentic Cambridge IELTS test materials
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
    - **Use authentic materials** like Cambridge IELTS tests
    - **Time yourself** to simulate real test conditions
    
    ### 🔧 Technical Details
    
    - **Backend**: Python FastAPI with machine learning models
    - **Frontend**: Streamlit for easy interaction
    - **Models**: Random Forest, Neural Networks, and rule-based scoring
    - **Data**: Cambridge IELTS test materials and official scoring criteria
    
    ### 📞 Support
    
    If you have any questions or need help, please don't hesitate to reach out. We're here to help you succeed in your IELTS journey!
    
    ---
    
    **Good luck with your IELTS preparation! 🍀**
    """)

if __name__ == "__main__":
    main()
