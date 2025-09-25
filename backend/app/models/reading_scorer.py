"""
IELTS Reading Test Scorer
Handles scoring and feedback for reading tests
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from .reading_test_data import ReadingTest, Question, QuestionType, get_reading_test

@dataclass
class UserAnswer:
    question_id: int
    answer: str
    time_taken: float = 0.0  # in seconds

@dataclass
class ReadingTestSubmission:
    test_id: int
    answers: List[UserAnswer]
    total_time_taken: float = 0.0  # in minutes

@dataclass
class ReadingTestResult:
    test_id: int
    user_answers: List[UserAnswer]
    total_questions: int
    correct_answers: int
    score: float  # out of 40
    band_score: float  # IELTS band score
    time_taken: float  # total time in minutes
    detailed_feedback: Dict[str, Any]
    question_analysis: List[Dict[str, Any]]

class ReadingScorer:
    """
    Scores IELTS Reading tests and provides detailed feedback
    """
    
    def __init__(self):
        # Official IELTS Reading band score conversion table (Academic)
        self.band_score_table = {
            40: 9.0, 39: 8.5, 38: 8.5, 37: 8.0, 36: 8.0,
            35: 7.5, 34: 7.5, 33: 7.0, 32: 7.0, 31: 7.0,
            30: 7.0, 29: 6.5, 28: 6.5, 27: 6.5, 26: 6.5,
            25: 6.0, 24: 6.0, 23: 6.0, 22: 6.0, 21: 6.0,
            20: 5.5, 19: 5.5, 18: 5.5, 17: 5.5, 16: 5.5,
            15: 5.0, 14: 5.0, 13: 5.0, 12: 5.0, 11: 5.0,
            10: 4.5, 9: 4.5, 8: 4.5, 7: 4.5, 6: 4.5,
            5: 4.0, 4: 4.0, 3: 3.5, 2: 3.0, 1: 2.5,
            0: 1.0
        }
    
    def score_reading_test(self, test_id: int, user_answers: List[UserAnswer], time_taken: float) -> ReadingTestResult:
        """
        Score a reading test and provide detailed feedback
        """
        test = get_reading_test(test_id)
        if not test:
            raise ValueError(f"Reading test with ID {test_id} not found")
        
        # Score the test
        correct_count = 0
        question_analysis = []
        
        for user_answer in user_answers:
            question = self._find_question(test, user_answer.question_id)
            if not question:
                continue
            
            is_correct = self._check_answer(question, user_answer.answer)
            if is_correct:
                correct_count += 1
            
            # Analyze the question
            analysis = {
                "question_id": question.id,
                "question_type": question.type.value,
                "user_answer": user_answer.answer,
                "correct_answer": question.correct_answer,
                "is_correct": is_correct,
                "explanation": question.explanation,
                "time_taken": user_answer.time_taken
            }
            question_analysis.append(analysis)
        
        # Calculate scores
        score = correct_count
        band_score = self.band_score_table.get(score, 1.0)  # Default to Band 1.0 for very low scores
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(
            test, correct_count, score, band_score, time_taken, question_analysis
        )
        
        return ReadingTestResult(
            test_id=test_id,
            user_answers=user_answers,
            total_questions=test.total_questions,
            correct_answers=correct_count,
            score=score,
            band_score=band_score,
            time_taken=time_taken,
            detailed_feedback=detailed_feedback,
            question_analysis=question_analysis
        )
    
    def _find_question(self, test: ReadingTest, question_id: int) -> Question:
        """Find a question by ID in the test"""
        for question in test.questions:
            if question.id == question_id:
                return question
        return None
    
    def _check_answer(self, question: Question, user_answer: str) -> bool:
        """Check if a user's answer is correct"""
        # Normalize answers for comparison
        correct_answer = str(question.correct_answer).strip().lower()
        user_answer = str(user_answer).strip().lower()
        
        # Handle different question types
        if question.type == QuestionType.TRUE_FALSE_NOT_GIVEN:
            # For True/False/Not Given questions, check for exact match
            return correct_answer == user_answer
        
        elif question.type == QuestionType.MULTIPLE_CHOICE:
            # For multiple choice, check if user selected the correct option
            return correct_answer == user_answer
        
        elif question.type in [QuestionType.SENTENCE_COMPLETION, QuestionType.SHORT_ANSWER]:
            # For completion questions, allow some flexibility
            return self._fuzzy_match(correct_answer, user_answer)
        
        else:
            # Default to exact match
            return correct_answer == user_answer
    
    def _fuzzy_match(self, correct: str, user: str) -> bool:
        """Fuzzy matching for completion questions"""
        # Remove common variations
        correct = correct.replace("the ", "").replace("a ", "").replace("an ", "")
        user = user.replace("the ", "").replace("a ", "").replace("an ", "")
        
        # Check for exact match
        if correct == user:
            return True
        
        # Check if user answer contains the correct answer
        if correct in user or user in correct:
            return True
        
        # Check for common synonyms (basic implementation)
        synonyms = {
            "increase": ["rise", "grow", "expand"],
            "decrease": ["reduce", "decline", "fall"],
            "important": ["significant", "crucial", "vital"],
            "difficult": ["challenging", "hard", "complex"]
        }
        
        for key, values in synonyms.items():
            if (correct == key and user in values) or (user == key and correct in values):
                return True
        
        return False
    
    def _generate_detailed_feedback(self, test: ReadingTest, correct_count: int, 
                                  score: float, band_score: float, time_taken: float, 
                                  question_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive feedback for the reading test"""
        
        # Calculate performance by question type
        type_performance = {}
        for analysis in question_analysis:
            q_type = analysis["question_type"]
            if q_type not in type_performance:
                type_performance[q_type] = {"correct": 0, "total": 0}
            type_performance[q_type]["total"] += 1
            if analysis["is_correct"]:
                type_performance[q_type]["correct"] += 1
        
        # Calculate accuracy by question type
        type_accuracy = {}
        for q_type, perf in type_performance.items():
            accuracy = (perf["correct"] / perf["total"]) * 100 if perf["total"] > 0 else 0
            type_accuracy[q_type] = {
                "accuracy": accuracy,
                "correct": perf["correct"],
                "total": perf["total"]
            }
        
        # Time analysis
        time_per_question = time_taken / test.total_questions if test.total_questions > 0 else 0
        time_efficiency = "Good" if time_per_question <= 1.5 else "Needs Improvement"
        
        # Overall performance assessment
        if band_score >= 7.0:
            performance_level = "Excellent"
            performance_feedback = "Outstanding performance! You demonstrate strong reading comprehension skills."
        elif band_score >= 6.0:
            performance_level = "Good"
            performance_feedback = "Good performance with room for improvement in some areas."
        elif band_score >= 5.0:
            performance_level = "Moderate"
            performance_feedback = "Moderate performance. Focus on improving reading strategies and vocabulary."
        else:
            performance_level = "Needs Improvement"
            performance_feedback = "Significant improvement needed. Focus on basic reading skills and vocabulary."
        
        # Generate specific recommendations
        recommendations = self._generate_recommendations(type_accuracy, band_score, time_efficiency)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for q_type, acc in type_accuracy.items():
            if acc["accuracy"] >= 80:
                strengths.append(f"Strong performance in {q_type.replace('_', ' ').title()} questions")
            elif acc["accuracy"] < 60:
                weaknesses.append(f"Need improvement in {q_type.replace('_', ' ').title()} questions")
        
        return {
            "overall_performance": {
                "level": performance_level,
                "feedback": performance_feedback,
                "band_score": band_score,
                "raw_score": f"{correct_count}/{test.total_questions}"
            },
            "time_analysis": {
                "total_time": f"{time_taken:.1f} minutes",
                "time_per_question": f"{time_per_question:.1f} minutes",
                "efficiency": time_efficiency,
                "recommendation": "Aim for 1.5 minutes per question" if time_per_question > 1.5 else "Good time management"
            },
            "question_type_performance": type_accuracy,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps(band_score, type_accuracy)
        }
    
    def _generate_recommendations(self, type_accuracy: Dict[str, Any], 
                                band_score: float, time_efficiency: str) -> List[str]:
        """Generate specific recommendations based on performance"""
        recommendations = []
        
        # Time management recommendations
        if time_efficiency == "Needs Improvement":
            recommendations.append("Practice time management - aim to spend no more than 1.5 minutes per question")
            recommendations.append("Learn to identify when to move on from difficult questions")
        
        # Question type specific recommendations
        for q_type, acc in type_accuracy.items():
            if acc["accuracy"] < 60:
                if q_type == "multiple_choice":
                    recommendations.append("Practice multiple choice questions - focus on eliminating wrong options")
                elif q_type == "true_false_not_given":
                    recommendations.append("Study True/False/Not Given questions - understand the difference between 'False' and 'Not Given'")
                elif q_type == "sentence_completion":
                    recommendations.append("Practice sentence completion - pay attention to grammar and word limits")
                elif q_type == "matching_headings":
                    recommendations.append("Improve matching headings skills - focus on identifying main ideas in paragraphs")
        
        # General recommendations based on band score
        if band_score < 6.0:
            recommendations.append("Build vocabulary - learn academic words and their meanings")
            recommendations.append("Practice reading comprehension - focus on understanding main ideas and details")
            recommendations.append("Improve reading speed - practice reading academic texts regularly")
        elif band_score < 7.0:
            recommendations.append("Focus on question types where you scored lower")
            recommendations.append("Practice with more challenging texts")
            recommendations.append("Work on inference and implied meaning questions")
        else:
            recommendations.append("Maintain your strong performance with regular practice")
            recommendations.append("Challenge yourself with more difficult texts")
        
        return recommendations
    
    def _generate_next_steps(self, band_score: float, type_accuracy: Dict[str, Any]) -> List[str]:
        """Generate next steps for improvement"""
        next_steps = []
        
        if band_score < 6.0:
            next_steps.extend([
                "Take more practice tests to build confidence",
                "Focus on vocabulary building exercises",
                "Practice with easier texts first, then gradually increase difficulty",
                "Work on basic reading comprehension strategies"
            ])
        elif band_score < 7.0:
            next_steps.extend([
                "Identify your weakest question types and practice them specifically",
                "Work on reading speed and time management",
                "Practice with authentic IELTS reading materials",
                "Focus on understanding implied meanings and inferences"
            ])
        else:
            next_steps.extend([
                "Continue practicing to maintain your high level",
                "Challenge yourself with more difficult texts",
                "Focus on achieving consistency across all question types",
                "Consider taking the actual IELTS test"
            ])
        
        return next_steps
