"""
Load Reading Tests into System
Loads the extracted Cambridge IELTS reading tests into our system
"""

import json
from pathlib import Path
from app.models.reading_test_data import ReadingTest, ReadingPassage, Question, QuestionType

def load_stepwells_test():
    """Load the stepwells reading test"""
    
    # Load the JSON file
    json_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/backend/stepwells_reading_test.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to our system format
    passages = []
    for passage_data in data["passages"]:
        passage = ReadingPassage(
            id=passage_data["id"],
            title=passage_data["title"],
            content=passage_data["content"],
            word_count=passage_data["word_count"],
            difficulty_level=passage_data["difficulty_level"],
            topic=passage_data["topic"]
        )
        passages.append(passage)
    
    questions = []
    for question_data in data["questions"]:
        # Convert question type string to enum
        question_type = QuestionType.MULTIPLE_CHOICE  # Default
        if question_data["type"] == "true_false_not_given":
            question_type = QuestionType.TRUE_FALSE_NOT_GIVEN
        elif question_data["type"] == "sentence_completion":
            question_type = QuestionType.SENTENCE_COMPLETION
        elif question_data["type"] == "short_answer":
            question_type = QuestionType.SHORT_ANSWER
        
        question = Question(
            id=question_data["id"],
            type=question_type,
            question_text=question_data["question_text"],
            correct_answer=question_data["correct_answer"],
            explanation=question_data["explanation"],
            options=question_data.get("options"),
            passage_reference=question_data.get("passage_reference")
        )
        questions.append(question)
    
    # Create reading test
    reading_test = ReadingTest(
        id=data["id"],
        title=data["title"],
        time_limit=data["time_limit"],
        total_questions=data["total_questions"],
        difficulty=data["difficulty"],
        passages=passages,
        questions=questions
    )
    
    return reading_test

def create_sample_reading_tests():
    """Create sample reading tests for the system"""
    
    # Load the stepwells test
    stepwells_test = load_stepwells_test()
    
    # Create a list of tests
    tests = [stepwells_test]
    
    return tests

def main():
    """Test loading reading tests"""
    print("🔄 Loading reading tests...")
    
    try:
        tests = create_sample_reading_tests()
        
        print(f"✅ Successfully loaded {len(tests)} reading tests:")
        
        for test in tests:
            print(f"  📚 {test.title}")
            print(f"     Passages: {len(test.passages)}")
            print(f"     Questions: {len(test.questions)}")
            print(f"     Time Limit: {test.time_limit} minutes")
            
            if test.passages:
                passage = test.passages[0]
                print(f"     Sample Passage: {passage.title} ({passage.word_count} words)")
                print(f"     Content Preview: {passage.content[:100]}...")
            
            print()
        
        return tests
        
    except Exception as e:
        print(f"❌ Error loading reading tests: {e}")
        return []

if __name__ == "__main__":
    main()
