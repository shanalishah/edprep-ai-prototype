"""
Reading Test Data Generator
Converts Cambridge IELTS PDF content into our system format
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from refined_ielts_parser import RefinedIELTSParser
from app.models.reading_test_data import ReadingTest, ReadingPassage, Question, QuestionType

class ReadingTestGenerator:
    """Generates reading tests from Cambridge IELTS PDFs"""
    
    def __init__(self):
        self.parser = RefinedIELTSParser()
        self.cambridge_path = Path("/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS")
    
    def convert_to_system_format(self, pdf_path: str) -> ReadingTest:
        """Convert Cambridge PDF to our system format"""
        result = self.parser.parse_cambridge_test_clean(pdf_path)
        
        # Convert passages
        passages = []
        for i, passage_data in enumerate(result["passages"], 1):
            passage = ReadingPassage(
                id=i,
                title=passage_data.title,
                content=passage_data.content,
                word_count=passage_data.word_count,
                difficulty_level=passage_data.difficulty.lower(),
                topic=passage_data.topic
            )
            passages.append(passage)
        
        # Convert questions
        questions = []
        question_id = 1
        
        for question_data in result["questions"]:
            # Determine question type based on content
            question_type = self._determine_question_type(question_data.question_text)
            
            question = Question(
                id=question_id,
                passage_id=question_data.passage_id,
                type=question_type,
                question_text=question_data.question_text,
                options=self._generate_sample_options(question_type),
                correct_answer=self._generate_sample_answer(question_type),
                explanation=f"Answer explanation for question {question_id}",
                passage_reference=f"Passage {question_data.passage_id}"
            )
            questions.append(question)
            question_id += 1
        
        # Create reading test
        reading_test = ReadingTest(
            id=1,  # Will be assigned properly in the system
            title=result["title"],
            time_limit=60,  # Standard IELTS reading time
            total_questions=len(questions),
            difficulty="medium",
            passages=passages,
            questions=questions
        )
        
        return reading_test
    
    def _determine_question_type(self, question_text: str) -> QuestionType:
        """Determine question type from question text"""
        question_lower = question_text.lower()
        
        if "choose" in question_lower and "letters" in question_lower:
            return QuestionType.MULTIPLE_CHOICE
        elif "true" in question_lower and "false" in question_lower:
            return QuestionType.TRUE_FALSE_NOT_GIVEN
        elif "complete" in question_lower and "sentence" in question_lower:
            return QuestionType.SENTENCE_COMPLETION
        elif "complete" in question_lower and "summary" in question_lower:
            return QuestionType.SUMMARY_COMPLETION
        elif "complete" in question_lower and "table" in question_lower:
            return QuestionType.TABLE_COMPLETION
        elif "complete" in question_lower and "flowchart" in question_lower:
            return QuestionType.FLOWCHART_COMPLETION
        elif "match" in question_lower and "heading" in question_lower:
            return QuestionType.MATCHING_HEADINGS
        elif "label" in question_lower and "diagram" in question_lower:
            return QuestionType.LABELING_DIAGRAM
        else:
            return QuestionType.SHORT_ANSWER
    
    def _generate_sample_options(self, question_type: QuestionType) -> List[str]:
        """Generate sample options for multiple choice questions"""
        if question_type == QuestionType.MULTIPLE_CHOICE:
            return [
                "Option A: First choice",
                "Option B: Second choice", 
                "Option C: Third choice",
                "Option D: Fourth choice"
            ]
        return None
    
    def _generate_sample_answer(self, question_type: QuestionType) -> str:
        """Generate sample correct answer"""
        if question_type == QuestionType.MULTIPLE_CHOICE:
            return "Option B: Second choice"
        elif question_type == QuestionType.TRUE_FALSE_NOT_GIVEN:
            return "True"
        elif question_type in [QuestionType.SENTENCE_COMPLETION, QuestionType.SHORT_ANSWER]:
            return "Sample answer"
        else:
            return "Sample answer"
    
    def generate_sample_tests(self, num_tests: int = 3) -> List[ReadingTest]:
        """Generate sample reading tests from Cambridge PDFs"""
        pdf_files = self.parser.cambridge_path.rglob("*.pdf")
        pdf_list = list(pdf_files)[:num_tests]
        
        tests = []
        for i, pdf_path in enumerate(pdf_list, 1):
            try:
                test = self.convert_to_system_format(str(pdf_path))
                test.id = i  # Assign proper ID
                tests.append(test)
                print(f"✅ Generated test {i}: {test.title}")
                print(f"   Passages: {len(test.passages)}, Questions: {len(test.questions)}")
            except Exception as e:
                print(f"❌ Error generating test from {pdf_path}: {e}")
        
        return tests
    
    def save_tests_to_json(self, tests: List[ReadingTest], output_path: str):
        """Save tests to JSON file"""
        tests_data = []
        
        for test in tests:
            test_data = {
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
                        "passage_id": question.passage_id,
                        "type": question.type.value,
                        "question_text": question.question_text,
                        "options": question.options,
                        "correct_answer": question.correct_answer,
                        "explanation": question.explanation,
                        "passage_reference": question.passage_reference
                    } for question in test.questions
                ]
            }
            tests_data.append(test_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tests_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(tests)} tests to {output_path}")

def main():
    """Generate sample reading tests"""
    generator = ReadingTestGenerator()
    
    print("🔄 Generating reading tests from Cambridge IELTS PDFs...")
    tests = generator.generate_sample_tests(2)  # Generate 2 tests
    
    if tests:
        print(f"\n📊 Generated {len(tests)} reading tests:")
        for test in tests:
            print(f"  - {test.title}: {len(test.passages)} passages, {len(test.questions)} questions")
        
        # Save to JSON
        output_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/backend/cambridge_reading_tests.json"
        generator.save_tests_to_json(tests, output_path)
        
        # Show sample content
        if tests[0].passages:
            sample_passage = tests[0].passages[0]
            print(f"\n📖 Sample Passage from {tests[0].title}:")
            print(f"Title: {sample_passage.title}")
            print(f"Word Count: {sample_passage.word_count}")
            print(f"Content Preview: {sample_passage.content[:300]}...")
    else:
        print("❌ No tests generated")

if __name__ == "__main__":
    main()
