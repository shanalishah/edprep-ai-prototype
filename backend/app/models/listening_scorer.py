"""
IELTS Listening Test Scorer
Handles scoring and feedback for listening tests
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from .listening_test_data import ListeningTest, ListeningQuestion, ListeningSection, get_listening_test

@dataclass
class UserListeningAnswer:
    """User's answer for a listening question"""
    question_id: int
    answer: str
    time_taken: float = 0.0  # in seconds

@dataclass
class ListeningTestSubmission:
    """Complete listening test submission"""
    test_id: int
    answers: List[UserListeningAnswer]
    total_time_taken: float = 0.0  # in minutes

@dataclass
class ListeningQuestionAnalysis:
    """Analysis of a single listening question"""
    question_id: int
    question_type: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str
    time_taken: float

@dataclass
class ListeningTestResult:
    """Complete listening test result"""
    test_id: int
    total_questions: int
    correct_answers: int
    score: int  # Raw score out of 40
    band_score: float  # IELTS band score
    time_taken: float
    detailed_feedback: Dict[str, Any]
    question_analysis: List[ListeningQuestionAnalysis]

class ListeningScorer:
    """
    Scores IELTS Listening tests and provides detailed feedback
    """
    
    def __init__(self):
        # Official IELTS Listening band score conversion table
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
    
    def score_listening_test(self, test_id: int, user_answers: List[UserListeningAnswer], time_taken: float) -> ListeningTestResult:
        """
        Score a listening test and provide detailed feedback
        """
        test = get_listening_test(test_id)
        if not test:
            raise ValueError(f"Listening test with ID {test_id} not found")
        
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
            analysis = ListeningQuestionAnalysis(
                question_id=question.id,
                question_type=question.type.value,
                user_answer=user_answer.answer,
                correct_answer=question.correct_answer,
                is_correct=is_correct,
                explanation=question.explanation or f"The correct answer is '{question.correct_answer}'.",
                time_taken=user_answer.time_taken
            )
            question_analysis.append(analysis)
        
        # Calculate scores
        score = correct_count
        band_score = self.band_score_table.get(score, 1.0)
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(
            test, correct_count, score, band_score, time_taken, question_analysis
        )
        
        return ListeningTestResult(
            test_id=test_id,
            total_questions=test.total_questions,
            correct_answers=correct_count,
            score=score,
            band_score=band_score,
            time_taken=time_taken,
            detailed_feedback=detailed_feedback,
            question_analysis=question_analysis
        )
    
    def _find_question(self, test: ListeningTest, question_id: int) -> ListeningQuestion:
        """Find a question by ID in the test"""
        for question in test.questions:
            if question.id == question_id:
                return question
        return None
    
    def _check_answer(self, question: ListeningQuestion, user_answer: str) -> bool:
        """Check if a user's answer is correct"""
        # Normalize answers for comparison
        correct_answer = str(question.correct_answer).strip().lower()
        user_answer = str(user_answer).strip().lower()
        
        # Check exact match first
        if correct_answer == user_answer:
            return True
        
        # Check alternative answers
        if question.alternative_answers:
            for alt_answer in question.alternative_answers:
                if str(alt_answer).strip().lower() == user_answer:
                    return True
        
        # Handle special cases for listening answers
        return self._fuzzy_match_listening(correct_answer, user_answer, question)
    
    def _fuzzy_match_listening(self, correct: str, user: str, question: ListeningQuestion) -> bool:
        """Fuzzy matching for listening answers with IELTS-specific rules"""
        # Remove common variations
        correct = correct.replace("the ", "").replace("a ", "").replace("an ", "")
        user = user.replace("the ", "").replace("a ", "").replace("an ", "")
        
        # Check for exact match after normalization
        if correct == user:
            return True
        
        # Handle common IELTS listening variations
        # Numbers: "7" vs "seven"
        if correct.isdigit() and user.isdigit():
            return correct == user
        
        # Handle alternative spellings (British vs American)
        spelling_variations = {
            "colour": "color",
            "favour": "favor",
            "centre": "center",
            "theatre": "theater"
        }
        
        for british, american in spelling_variations.items():
            if (correct == british and user == american) or (correct == american and user == british):
                return True
        
        # Handle contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "it's": "it is"
        }
        
        for contraction, full_form in contractions.items():
            if (correct == contraction and user == full_form) or (correct == full_form and user == contraction):
                return True
        
        return False
    
    def _generate_detailed_feedback(self, test: ListeningTest, correct_count: int, score: int, 
                                  band_score: float, time_taken: float, 
                                  question_analysis: List[ListeningQuestionAnalysis]) -> Dict[str, Any]:
        """Generate detailed feedback for the listening test"""
        
        # Calculate section-wise performance
        section_performance = self._calculate_section_performance(question_analysis)
        
        # Calculate question type performance
        type_performance = self._calculate_type_performance(question_analysis)
        
        # Generate overall performance assessment
        overall_performance = self._assess_overall_performance(band_score, correct_count, test.total_questions)
        
        # Generate time analysis
        time_analysis = self._analyze_time_performance(time_taken, test.time_limit)
        
        # Generate strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(section_performance, type_performance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(band_score, section_performance, type_performance)
        
        # Generate next steps
        next_steps = self._generate_next_steps(band_score, type_performance)
        
        return {
            "overall_performance": overall_performance,
            "time_analysis": time_analysis,
            "section_performance": section_performance,
            "question_type_performance": type_performance,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "next_steps": next_steps
        }
    
    def _calculate_section_performance(self, question_analysis: List[ListeningQuestionAnalysis]) -> Dict[str, Any]:
        """Calculate performance by section"""
        section_stats = {}
        
        for analysis in question_analysis:
            # Extract section from question_id (assuming sections are 1-10, 11-20, 21-30, 31-40)
            if analysis.question_id <= 10:
                section = "Section 1"
            elif analysis.question_id <= 20:
                section = "Section 2"
            elif analysis.question_id <= 30:
                section = "Section 3"
            else:
                section = "Section 4"
            
            if section not in section_stats:
                section_stats[section] = {"correct": 0, "total": 0}
            
            section_stats[section]["total"] += 1
            if analysis.is_correct:
                section_stats[section]["correct"] += 1
        
        # Calculate accuracy for each section
        for section in section_stats:
            stats = section_stats[section]
            stats["accuracy"] = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        return section_stats
    
    def _calculate_type_performance(self, question_analysis: List[ListeningQuestionAnalysis]) -> Dict[str, Any]:
        """Calculate performance by question type"""
        type_stats = {}
        
        for analysis in question_analysis:
            q_type = analysis.question_type
            if q_type not in type_stats:
                type_stats[q_type] = {"correct": 0, "total": 0}
            
            type_stats[q_type]["total"] += 1
            if analysis.is_correct:
                type_stats[q_type]["correct"] += 1
        
        # Calculate accuracy for each type
        for q_type in type_stats:
            stats = type_stats[q_type]
            stats["accuracy"] = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        return type_stats
    
    def _assess_overall_performance(self, band_score: float, correct_count: int, total_questions: int) -> Dict[str, Any]:
        """Assess overall performance level"""
        percentage = (correct_count / total_questions) * 100
        
        if band_score >= 7.0:
            level = "Excellent"
            feedback = "Outstanding listening skills. You demonstrate excellent comprehension of both general and academic English."
        elif band_score >= 6.0:
            level = "Good"
            feedback = "Good listening skills with room for improvement in some areas."
        elif band_score >= 5.0:
            level = "Modest"
            feedback = "Modest listening skills. Focus on improving comprehension and note-taking."
        elif band_score >= 4.0:
            level = "Limited"
            feedback = "Limited listening skills. Significant improvement needed in basic comprehension."
        else:
            level = "Very Limited"
            feedback = "Very limited listening skills. Focus on basic vocabulary and simple conversations."
        
        return {
            "level": level,
            "feedback": feedback,
            "band_score": band_score,
            "raw_score": f"{correct_count}/{total_questions}",
            "percentage": round(percentage, 1)
        }
    
    def _analyze_time_performance(self, time_taken: float, time_limit: int) -> Dict[str, Any]:
        """Analyze time management performance"""
        time_percentage = (time_taken / time_limit) * 100
        
        if time_percentage <= 80:
            efficiency = "Excellent"
            recommendation = "Great time management. You completed the test efficiently."
        elif time_percentage <= 100:
            efficiency = "Good"
            recommendation = "Good time management. You used the time effectively."
        else:
            efficiency = "Needs Improvement"
            recommendation = "Work on time management. Practice completing tests within the time limit."
        
        return {
            "total_time": f"{time_taken:.1f} minutes",
            "time_limit": f"{time_limit} minutes",
            "time_percentage": round(time_percentage, 1),
            "efficiency": efficiency,
            "recommendation": recommendation
        }
    
    def _identify_strengths_weaknesses(self, section_performance: Dict, type_performance: Dict) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze section performance
        for section, stats in section_performance.items():
            if stats["accuracy"] >= 80:
                strengths.append(f"Strong performance in {section}")
            elif stats["accuracy"] < 50:
                weaknesses.append(f"Need improvement in {section}")
        
        # Analyze question type performance
        for q_type, stats in type_performance.items():
            if stats["accuracy"] >= 80:
                strengths.append(f"Excellent {q_type.replace('_', ' ')} skills")
            elif stats["accuracy"] < 50:
                weaknesses.append(f"Need improvement in {q_type.replace('_', ' ')} questions")
        
        return strengths, weaknesses
    
    def _generate_recommendations(self, band_score: float, section_performance: Dict, type_performance: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if band_score < 5.0:
            recommendations.extend([
                "Build basic vocabulary - learn common words used in daily conversations",
                "Practice listening to simple English conversations and news",
                "Focus on understanding main ideas before details"
            ])
        
        if band_score < 6.0:
            recommendations.extend([
                "Practice note-taking while listening",
                "Work on understanding different accents",
                "Practice listening to academic lectures and presentations"
            ])
        
        if band_score < 7.0:
            recommendations.extend([
                "Practice listening to complex academic content",
                "Work on understanding implied meanings and attitudes",
                "Practice with authentic IELTS listening materials"
            ])
        
        # Section-specific recommendations
        for section, stats in section_performance.items():
            if stats["accuracy"] < 60:
                if "Section 1" in section:
                    recommendations.append("Practice listening to everyday conversations and form-filling")
                elif "Section 2" in section:
                    recommendations.append("Practice listening to informational talks and announcements")
                elif "Section 3" in section:
                    recommendations.append("Practice listening to academic discussions and tutorials")
                elif "Section 4" in section:
                    recommendations.append("Practice listening to academic lectures and presentations")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_next_steps(self, band_score: float, type_performance: Dict) -> List[str]:
        """Generate next steps for improvement"""
        next_steps = []
        
        if band_score < 5.0:
            next_steps.extend([
                "Start with basic listening exercises and simple conversations",
                "Focus on building vocabulary first",
                "Practice with easier listening materials before attempting IELTS level"
            ])
        else:
            next_steps.extend([
                "Continue practicing with authentic IELTS listening tests",
                "Work on your weakest question types",
                "Practice listening to different English accents",
                "Take regular practice tests to track your progress"
            ])
        
        return next_steps
