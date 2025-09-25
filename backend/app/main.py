from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import os
from datetime import datetime
from dotenv import load_dotenv

# Import only lightweight models for deployment
from .models.essay_scorer import EssayScorer
from .models.feedback_generator import FeedbackGenerator

# Try to import ML models, but don't fail if they don't exist
try:
    from .models.ml_essay_scorer import MLEssayScorer
    from .models.production_essay_scorer import ProductionEssayScorer
    from .models.realistic_ielts_scorer import RealisticIELTSScorer
    from .models.optimized_semantic_analyzer import OptimizedSemanticAnalyzer
    from .models.optimized_feedback_generator import OptimizedFeedbackGenerator
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML models not available: {e}")
    ML_MODELS_AVAILABLE = False
# from .models.reading_test_data import get_reading_test, get_all_reading_tests, ReadingTest
# from .models.reading_scorer import ReadingScorer, UserAnswer, ReadingTestResult

# Import listening test components
from .models.listening_test_data import get_listening_test, get_all_listening_tests
from .models.listening_scorer import ListeningScorer, UserListeningAnswer, ListeningTestSubmission

# Load environment variables
load_dotenv()

app = FastAPI(
    title="EdPrep AI - IELTS Writing Assessment",
    description="AI-powered IELTS writing assessment and feedback system",
    version="1.0.0"
)

# CORS middleware - Allow all localhost ports for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001", 
        "http://localhost:3002", "http://127.0.0.1:3002",
        "http://localhost:3003", "http://127.0.0.1:3003",
        "http://localhost:3004", "http://127.0.0.1:3004",
        "http://localhost:3005", "http://127.0.0.1:3005"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files for audio
cambridge_audio_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/Academic/Cambridge IELTS 10 with Answers Academic [www.luckyielts.com]"
if os.path.exists(cambridge_audio_path):
    app.mount("/audio", StaticFiles(directory=cambridge_audio_path), name="audio")

# Initialize models - lightweight version for deployment
essay_scorer = EssayScorer()
feedback_generator = FeedbackGenerator()

# Initialize ML models only if available
if ML_MODELS_AVAILABLE:
    try:
        ml_essay_scorer = MLEssayScorer()
        production_essay_scorer = ProductionEssayScorer()
        realistic_ielts_scorer = RealisticIELTSScorer()
        optimized_semantic_analyzer = OptimizedSemanticAnalyzer()
        optimized_feedback_generator = OptimizedFeedbackGenerator()
        print("✅ ML models loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load ML models: {e}")
        ML_MODELS_AVAILABLE = False
else:
    print("⚠️ Using lightweight mode - ML models not available")
# reading_scorer = ReadingScorer()  # Reading test scorer

class EssaySubmission(BaseModel):
    prompt: str
    essay: str
    task_type: Optional[str] = "Task 2"  # Task 1 or Task 2

# class UserAnswerSubmission(BaseModel):
#     question_id: int
#     answer: str
#     time_taken: Optional[float] = 0.0

# class ReadingTestSubmission(BaseModel):
#     test_id: int
#     answers: List[UserAnswerSubmission]
#     total_time_taken: float  # in minutes

class ScoringResponse(BaseModel):
    task_achievement: float
    coherence_cohesion: float
    lexical_resource: float
    grammatical_range: float
    overall_band_score: float
    feedback: str
    suggestions: List[str]

@app.get("/")
async def root():
    return {"message": "EdPrep AI - IELTS Writing Assessment API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "EdPrep AI Backend"}

@app.get("/model-status")
async def get_model_status():
    """
    Get information about the scoring models
    """
    # Check which model is available
    if ML_MODELS_AVAILABLE and 'production_essay_scorer' in locals() and production_essay_scorer.is_loaded:
        return {
            "production_model_loaded": True,
            "official_ielts_model_loaded": True,
            "scoring_method": "Production ML (Advanced Random Forest) + Official IELTS Criteria",
            "model_path": str(production_essay_scorer.models_dir),
            "features": "535 advanced features + TF-IDF + Official IELTS Band Descriptors"
        }
    elif ML_MODELS_AVAILABLE and 'ml_essay_scorer' in locals() and ml_essay_scorer.is_loaded:
        return {
            "production_model_loaded": False,
            "basic_ml_model_loaded": True,
            "official_ielts_model_loaded": True,
            "scoring_method": "Basic ML (Random Forest) + Official IELTS Criteria",
            "model_path": str(ml_essay_scorer.models_dir),
            "features": "24 basic features + Official IELTS Band Descriptors"
        }
    else:
        return {
            "production_model_loaded": False,
            "basic_ml_model_loaded": False,
            "official_ielts_model_loaded": True,
            "scoring_method": "Official IELTS Criteria (Based on Official Band Descriptors) - Lightweight Mode",
            "model_path": None,
            "features": "Official IELTS Band Descriptors + Rule-based scoring"
        }

@app.post("/assess", response_model=ScoringResponse)
async def assess_essay(essay_data: EssaySubmission):
    """
    Assess an IELTS essay using available models (lightweight mode for deployment)
    """
    try:
        # Use available scorer (ML if available, otherwise rule-based)
        if ML_MODELS_AVAILABLE and 'realistic_ielts_scorer' in locals():
            scores = realistic_ielts_scorer.score_essay_realistic(
                essay=essay_data.essay,
                prompt=essay_data.prompt,
                task_type=essay_data.task_type
            )
        else:
            # Fallback to rule-based scoring
            scores = essay_scorer.score_essay(
                essay=essay_data.essay,
                prompt=essay_data.prompt,
                task_type=essay_data.task_type
            )
        
        # Generate detailed feedback
        feedback = feedback_generator.generate_feedback(
            prompt=essay_data.prompt,
            essay=essay_data.essay,
            scores=scores,
            task_type=essay_data.task_type
        )
        
        # Add realistic assessment information
        word_count = len(essay_data.essay.split())
        feedback_text = feedback.get('feedback', feedback.get('detailed_feedback', 'No feedback available'))
        enhanced_feedback = f"{feedback_text}\n\n**Realistic Assessment:** This score is based on strict IELTS criteria and official band descriptors.\n**Word Count:** {word_count} words"
        
        if word_count < 250 and essay_data.task_type == "Task 2":
            enhanced_feedback += f"\n**⚠️ Warning:** Task 2 requires at least 250 words. Your essay is {250 - word_count} words short."
        elif word_count < 150 and essay_data.task_type == "Task 1":
            enhanced_feedback += f"\n**⚠️ Warning:** Task 1 requires at least 150 words. Your essay is {150 - word_count} words short."
        
        return ScoringResponse(
            task_achievement=scores["task_achievement"],
            coherence_cohesion=scores["coherence_cohesion"],
            lexical_resource=scores["lexical_resource"],
            grammatical_range=scores["grammatical_range"],
            overall_band_score=scores["overall_band_score"],
            feedback=enhanced_feedback,
            suggestions=feedback.get("suggestions", ["Continue practicing to improve your writing skills"])
        )
        
    except Exception as e:
        # Fallback to basic scorer if quality scorer fails
        try:
            fallback_scores = essay_scorer.score_essay(
                prompt=essay_data.prompt,
                essay=essay_data.essay,
                task_type=essay_data.task_type
            )
            
            feedback = feedback_generator.generate_feedback(
                prompt=essay_data.prompt,
                essay=essay_data.essay,
                scores=fallback_scores,
                task_type=essay_data.task_type
            )
            
            return ScoringResponse(
                task_achievement=fallback_scores["task_achievement"],
                coherence_cohesion=fallback_scores["coherence_cohesion"],
                lexical_resource=fallback_scores["lexical_resource"],
                grammatical_range=fallback_scores["grammatical_range"],
                overall_band_score=fallback_scores["overall_band_score"],
                feedback=feedback["feedback"] + "\n\n**Note:** Using fallback scoring due to system issue.",
                suggestions=feedback.get("suggestions", ["Continue practicing to improve your writing skills"]) + ["Manual review recommended due to system issue"]
            )
        except Exception as fallback_error:
            raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}. Fallback also failed: {str(fallback_error)}")

@app.post("/toggle-strict-mode")
async def toggle_strict_mode():
    """
    Toggle between strict and balanced evaluation modes
    """
    hybrid_ielts_scorer.strict_mode = not hybrid_ielts_scorer.strict_mode
    
    # Update weights based on mode
    if hybrid_ielts_scorer.strict_mode:
        hybrid_ielts_scorer.weights = {
            "official_criteria": 0.7,  # Stricter: more weight on official criteria
            "ml_prediction": 0.3
        }
        mode = "Strict (70% Official IELTS + 30% ML)"
    else:
        hybrid_ielts_scorer.weights = {
            "official_criteria": 0.4,  # Balanced: more weight on ML
            "ml_prediction": 0.6
        }
        mode = "Balanced (40% Official IELTS + 60% ML)"
    
    return {
        "strict_mode": hybrid_ielts_scorer.strict_mode,
        "current_mode": mode,
        "weights": hybrid_ielts_scorer.weights
    }

@app.post("/test-hybrid-weights")
async def test_hybrid_weights(essay_data: EssaySubmission, rubric_weight: float = 0.5):
    """
    Test hybrid scoring with specific rubric weight
    """
    try:
        # Use hybrid essay scorer with specific weight
        result = hybrid_essay_scorer.score_essay_hybrid(
            essay=essay_data.essay,
            prompt=essay_data.prompt,
            task_type=essay_data.task_type,
            force_rubric_weight=rubric_weight
        )
        
        # Generate feedback
        feedback = feedback_generator.generate_feedback(
            prompt=essay_data.prompt,
            essay=essay_data.essay,
            scores=result["scores"],
            task_type=essay_data.task_type
        )
        
        return {
            "scores": result["scores"],
            "scoring_details": result["scoring_details"],
            "individual_scores": result["individual_scores"],
            "feedback": feedback["feedback"],
            "suggestions": feedback["suggestions"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid testing failed: {str(e)}")

@app.get("/hybrid-performance")
async def get_hybrid_performance():
    """
    Get performance summary of hybrid scorer
    """
    try:
        summary = hybrid_essay_scorer.get_performance_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance summary failed: {str(e)}")

@app.get("/quality-metrics")
async def get_quality_metrics():
    """
    Get quality metrics and monitoring data
    """
    try:
        metrics = quality_essay_scorer.get_performance_metrics()
        return {
            "quality_metrics": metrics,
            "model_status": "operational" if quality_essay_scorer.is_loaded else "degraded",
            "quality_thresholds": quality_essay_scorer.quality_thresholds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality metrics: {str(e)}")

@app.post("/assess-optimized", response_model=ScoringResponse)
async def assess_essay_optimized(essay_data: EssaySubmission):
    """
    Assess an IELTS essay using optimized semantic analysis and personalized feedback
    """
    try:
        # Get realistic scores
        scores = realistic_ielts_scorer.score_essay_realistic(
            essay=essay_data.essay,
            prompt=essay_data.prompt,
            task_type=essay_data.task_type
        )
        
        # Generate optimized personalized feedback
        optimized_feedback = optimized_feedback_generator.generate_optimized_feedback(
            essay=essay_data.essay,
            prompt=essay_data.prompt,
            task_type=essay_data.task_type,
            scores=scores
        )
        
        # Create comprehensive feedback
        word_count = len(essay_data.essay.split())
        enhanced_feedback = f"{optimized_feedback['overall_assessment']}\n\n"
        
        # Add criterion-specific feedback
        for criterion, feedback in optimized_feedback['criterion_feedback'].items():
            criterion_name = criterion.replace('_', ' ').title()
            enhanced_feedback += f"**{criterion_name}:** {feedback}\n\n"
        
        # Add strengths and weaknesses
        if optimized_feedback['strengths']:
            enhanced_feedback += f"**Strengths:**\n" + "\n".join([f"• {strength}" for strength in optimized_feedback['strengths']]) + "\n\n"
        
        if optimized_feedback['weaknesses']:
            enhanced_feedback += f"**Areas for Improvement:**\n" + "\n".join([f"• {weakness}" for weakness in optimized_feedback['weaknesses']]) + "\n\n"
        
        # Add semantic insights
        insights = optimized_feedback['semantic_insights']
        enhanced_feedback += f"**Writing Style Analysis:** {insights['writing_style']}\n\n"
        enhanced_feedback += f"**Language Sophistication:** {insights['language_sophistication']}\n\n"
        enhanced_feedback += f"**Coherence Quality:** {insights['coherence_quality']}\n\n"
        enhanced_feedback += f"**Academic Readiness:** {insights['academic_readiness']}\n\n"
        
        # Add improvement priorities
        if optimized_feedback['improvement_priority']:
            enhanced_feedback += f"**Improvement Priorities:**\n" + "\n".join(optimized_feedback['improvement_priority']) + "\n\n"
        
        # Add word count warning
        if word_count < 250 and essay_data.task_type == "Task 2":
            enhanced_feedback += f"**⚠️ CRITICAL:** Task 2 requires at least 250 words. Your essay is {250 - word_count} words short.\n\n"
        elif word_count < 150 and essay_data.task_type == "Task 1":
            enhanced_feedback += f"**⚠️ CRITICAL:** Task 1 requires at least 150 words. Your essay is {150 - word_count} words short.\n\n"
        
        enhanced_feedback += f"**Word Count:** {word_count} words"
        
        return ScoringResponse(
            task_achievement=scores["task_achievement"],
            coherence_cohesion=scores["coherence_cohesion"],
            lexical_resource=scores["lexical_resource"],
            grammatical_range=scores["grammatical_range"],
            overall_band_score=scores["overall_band_score"],
            feedback=enhanced_feedback,
            suggestions=optimized_feedback['specific_suggestions']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimized assessment failed: {str(e)}")

@app.post("/assess-realistic", response_model=ScoringResponse)
async def assess_essay_realistic(essay_data: EssaySubmission):
    """
    Assess an IELTS essay using ONLY the realistic, strict scorer
    """
    try:
        scores = realistic_ielts_scorer.score_essay_realistic(
            essay=essay_data.essay,
            prompt=essay_data.prompt,
            task_type=essay_data.task_type
        )
        
        # Generate detailed feedback
        feedback = feedback_generator.generate_feedback(
            prompt=essay_data.prompt,
            essay=essay_data.essay,
            scores=scores,
            task_type=essay_data.task_type
        )
        
        word_count = len(essay_data.essay.split())
        feedback_text = feedback.get('feedback', feedback.get('detailed_feedback', 'No feedback available'))
        enhanced_feedback = f"{feedback_text}\n\n**🎯 Strict IELTS Assessment:** This score uses authentic IELTS criteria and official band descriptors.\n**Word Count:** {word_count} words"
        
        if word_count < 250 and essay_data.task_type == "Task 2":
            enhanced_feedback += f"\n**⚠️ CRITICAL:** Task 2 requires at least 250 words. Your essay is {250 - word_count} words short. This significantly impacts your score."
        elif word_count < 150 and essay_data.task_type == "Task 1":
            enhanced_feedback += f"\n**⚠️ CRITICAL:** Task 1 requires at least 150 words. Your essay is {150 - word_count} words short. This significantly impacts your score."
        
        return ScoringResponse(
            task_achievement=scores["task_achievement"],
            coherence_cohesion=scores["coherence_cohesion"],
            lexical_resource=scores["lexical_resource"],
            grammatical_range=scores["grammatical_range"],
            overall_band_score=scores["overall_band_score"],
            feedback=enhanced_feedback,
            suggestions=feedback.get("suggestions", ["Continue practicing to improve your writing skills"])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Realistic assessment failed: {str(e)}")

# ==================== READING TEST ENDPOINTS ====================

# Import reading test components
from app.models.reading_test_data import get_reading_test, get_all_reading_tests
from app.models.reading_scorer import ReadingScorer, UserAnswer, ReadingTestSubmission

# Initialize reading scorer
reading_scorer = ReadingScorer()

# Initialize listening scorer
listening_scorer = ListeningScorer()

@app.get("/reading-tests")
async def get_reading_tests():
    """
    Get all available reading tests
    """
    try:
        tests = get_all_reading_tests()
        return {
            "tests": [
                {
                    "id": test.id,
                    "title": test.title,
                    "time_limit": test.time_limit,
                    "total_questions": test.total_questions,
                    "difficulty": test.difficulty,
                    "passages": [
                        {
                            "id": passage.id,
                            "title": passage.title,
                            "word_count": passage.word_count,
                            "difficulty_level": passage.difficulty_level,
                            "topic": passage.topic
                        } for passage in test.passages
                    ]
                } for test in tests
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reading tests: {str(e)}")

@app.get("/reading-tests/{test_id}")
async def get_reading_test_details(test_id: int):
    """
    Get detailed information about a specific reading test
    """
    try:
        test = get_reading_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Reading test with ID {test_id} not found")
        
        return {
            "id": test.id,
            "title": test.title,
            "time_limit": test.time_limit,
            "total_questions": test.total_questions,
            "difficulty": test.difficulty,
            "passages": [
                {
                    "id": passage.id,
                    "title": passage.title,
                    "content": passage.content,
                    "word_count": passage.word_count,
                    "difficulty_level": passage.difficulty_level,
                    "topic": passage.topic
                } for passage in test.passages
            ],
            "questions": [
                {
                    "id": question.id,
                    "type": question.type.value,
                    "question_text": question.question_text,
                    "options": question.options,
                    "passage_reference": question.passage_reference
                } for question in test.questions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reading test: {str(e)}")

@app.get("/reading-tests/{test_id}/passage/{passage_id}")
async def get_reading_passage(test_id: int, passage_id: int):
    """
    Get a specific passage from a reading test
    """
    try:
        test = get_reading_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Reading test with ID {test_id} not found")
        
        passage = None
        for p in test.passages:
            if p.id == passage_id:
                passage = p
                break
        
        if not passage:
            raise HTTPException(status_code=404, detail=f"Passage with ID {passage_id} not found in test {test_id}")
        
        return {
            "id": passage.id,
            "title": passage.title,
            "content": passage.content,
            "word_count": passage.word_count,
            "difficulty_level": passage.difficulty_level,
            "topic": passage.topic
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get passage: {str(e)}")

@app.get("/reading-tests/{test_id}/questions")
async def get_reading_questions(test_id: int):
    """
    Get all questions for a reading test
    """
    try:
        test = get_reading_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Reading test with ID {test_id} not found")
        
        return {
            "test_id": test_id,
            "questions": [
                {
                    "id": question.id,
                    "type": question.type.value,
                    "question_text": question.question_text,
                    "options": question.options,
                    "passage_reference": question.passage_reference
                } for question in test.questions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get questions: {str(e)}")

@app.post("/reading-tests/submit")
async def submit_reading_test(submission: ReadingTestSubmission):
    """
    Submit answers for a reading test and get results
    """
    try:
        # Convert submission to UserAnswer objects
        user_answers = [
            UserAnswer(
                question_id=answer.question_id,
                answer=answer.answer,
                time_taken=answer.time_taken
            ) for answer in submission.answers
        ]
        
        # Score the test
        result = reading_scorer.score_reading_test(
            test_id=submission.test_id,
            user_answers=user_answers,
            time_taken=submission.total_time_taken
        )
        
        return {
            "test_id": result.test_id,
            "total_questions": result.total_questions,
            "correct_answers": result.correct_answers,
            "score": result.score,
            "band_score": result.band_score,
            "time_taken": result.time_taken,
            "detailed_feedback": result.detailed_feedback,
            "question_analysis": result.question_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to score reading test: {str(e)}")

@app.get("/reading-tests/{test_id}/answers")
async def get_reading_test_answers(test_id: int):
    """
    Get correct answers for a reading test (for review purposes)
    """
    try:
        test = get_reading_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Reading test with ID {test_id} not found")
        
        return {
            "test_id": test_id,
            "answers": [
                {
                    "question_id": question.id,
                    "correct_answer": question.correct_answer,
                    "explanation": question.explanation,
                    "type": question.type.value
                } for question in test.questions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answers: {str(e)}")

# ==================== LISTENING TEST ENDPOINTS ====================

@app.get("/listening-tests")
async def get_listening_tests():
    """
    Get all available listening tests
    """
    try:
        tests = get_all_listening_tests()
        return {
            "tests": [
                {
                    "id": test.id,
                    "title": test.title,
                    "test_number": test.test_number,
                    "time_limit": test.time_limit,
                    "total_questions": test.total_questions,
                    "difficulty": test.difficulty,
                    "audio_tracks": [
                        {
                            "id": track.id,
                            "section": track.section.value,
                            "track_number": track.track_number,
                            "description": track.description,
                            "duration_seconds": track.duration_seconds
                        } for track in test.audio_tracks
                    ]
                } for test in tests
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get listening tests: {str(e)}")

@app.get("/listening-tests/{test_id}")
async def get_listening_test_details(test_id: int):
    """
    Get detailed information about a specific listening test
    """
    try:
        test = get_listening_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Listening test with ID {test_id} not found")
        
        return {
            "id": test.id,
            "title": test.title,
            "test_number": test.test_number,
            "time_limit": test.time_limit,
            "total_questions": test.total_questions,
            "difficulty": test.difficulty,
            "instructions": test.instructions,
            "audio_tracks": [
                {
                    "id": track.id,
                    "section": track.section.value,
                    "track_number": track.track_number,
                    "file_path": track.file_path,
                    "duration_seconds": track.duration_seconds,
                    "description": track.description
                } for track in test.audio_tracks
            ],
            "questions": [
                {
                    "id": question.id,
                    "section": question.section.value,
                    "question_number": question.question_number,
                    "type": question.type.value,
                    "question_text": question.question_text,
                    "word_limit": question.word_limit,
                    "context": question.context,
                    "options": question.options
                } for question in test.questions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get listening test: {str(e)}")

@app.get("/listening-tests/{test_id}/audio/{track_id}")
async def get_listening_audio(test_id: int, track_id: int):
    """
    Get audio track information for a listening test
    """
    try:
        test = get_listening_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Listening test with ID {test_id} not found")
        
        track = None
        for t in test.audio_tracks:
            if t.id == track_id:
                track = t
                break
        
        if not track:
            raise HTTPException(status_code=404, detail=f"Audio track with ID {track_id} not found in test {test_id}")
        
        return {
            "id": track.id,
            "section": track.section.value,
            "track_number": track.track_number,
            "file_path": track.file_path,
            "duration_seconds": track.duration_seconds,
            "description": track.description
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audio track: {str(e)}")

@app.get("/listening-tests/{test_id}/questions")
async def get_listening_questions(test_id: int):
    """
    Get all questions for a listening test
    """
    try:
        test = get_listening_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Listening test with ID {test_id} not found")
        
        return {
            "test_id": test_id,
            "questions": [
                {
                    "id": question.id,
                    "section": question.section.value,
                    "question_number": question.question_number,
                    "type": question.type.value,
                    "question_text": question.question_text,
                    "word_limit": question.word_limit,
                    "context": question.context,
                    "options": question.options
                } for question in test.questions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get questions: {str(e)}")

@app.post("/listening-tests/submit")
async def submit_listening_test(submission: ListeningTestSubmission):
    """
    Submit answers for a listening test and get results
    """
    try:
        # Convert submission to UserListeningAnswer objects
        user_answers = [
            UserListeningAnswer(
                question_id=answer.question_id,
                answer=answer.answer,
                time_taken=answer.time_taken
            ) for answer in submission.answers
        ]
        
        # Score the test
        result = listening_scorer.score_listening_test(
            test_id=submission.test_id,
            user_answers=user_answers,
            time_taken=submission.total_time_taken
        )
        
        return {
            "test_id": result.test_id,
            "total_questions": result.total_questions,
            "correct_answers": result.correct_answers,
            "score": result.score,
            "band_score": result.band_score,
            "time_taken": result.time_taken,
            "detailed_feedback": result.detailed_feedback,
            "question_analysis": result.question_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to score listening test: {str(e)}")

@app.get("/listening-tests/{test_id}/answers")
async def get_listening_test_answers(test_id: int):
    """
    Get correct answers for a listening test (for review purposes)
    """
    try:
        test = get_listening_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail=f"Listening test with ID {test_id} not found")
        
        return {
            "test_id": test_id,
            "answers": [
                {
                    "question_id": question.id,
                    "correct_answer": question.correct_answer,
                    "alternative_answers": question.alternative_answers,
                    "explanation": question.explanation,
                    "type": question.type.value
                } for question in test.questions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answers: {str(e)}")

@app.get("/sample-prompts")
async def get_sample_prompts():
    """
    Get sample IELTS writing prompts for testing
    """
    sample_prompts = [
        {
            "id": 1,
            "task_type": "Task 2",
            "prompt": "Some people think that the best way to reduce crime is to give longer prison sentences. Others, however, believe there are better alternative ways of reducing crime. Discuss both views and give your opinion.",
            "category": "Crime and Punishment"
        },
        {
            "id": 2,
            "task_type": "Task 2", 
            "prompt": "In many countries, more and more young people are leaving school and unable to find jobs after graduation. What problems do you think youth unemployment will cause to the individual and society? Give reasons and make suggestions.",
            "category": "Education and Employment"
        },
        {
            "id": 3,
            "task_type": "Task 2",
            "prompt": "Some people believe that technology has made our lives more complicated, while others think it has made our lives easier. Discuss both views and give your opinion.",
            "category": "Technology"
        }
    ]
    return {"prompts": sample_prompts}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


