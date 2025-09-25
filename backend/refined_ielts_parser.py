"""
Refined Cambridge IELTS Parser
Extracts clean reading passages, questions, and answers
"""

import fitz  # pymupdf
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class CleanReadingPassage:
    title: str
    content: str
    word_count: int
    difficulty: str
    topic: str
    passage_number: int

@dataclass
class CleanReadingQuestion:
    id: int
    passage_id: int
    type: str
    question_text: str
    options: List[str] = None
    correct_answer: str = ""
    explanation: str = ""

class RefinedIELTSParser:
    """Refined parser for clean Cambridge IELTS content"""
    
    def __init__(self):
        self.cambridge_path = Path("/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS")
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers
        text = re.sub(r'www\.luckyielts\.com', '', text)
        text = re.sub(r'Cambridge English', '', text)
        text = re.sub(r'OFFICIAL', '', text)
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def extract_reading_passages_clean(self, pdf_path: str) -> List[CleanReadingPassage]:
        """Extract clean reading passages"""
        try:
            doc = fitz.open(pdf_path)
            passages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Look for reading passage markers
                if re.search(r'READING\s*PASSAGE\s*\d+', text, re.IGNORECASE):
                    # Extract passage content
                    lines = text.split('\n')
                    passage_content = []
                    in_passage = False
                    passage_title = ""
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check for passage start
                        if re.search(r'READING\s*PASSAGE\s*(\d+)', line, re.IGNORECASE):
                            match = re.search(r'READING\s*PASSAGE\s*(\d+)', line, re.IGNORECASE)
                            if match:
                                passage_title = f"Reading Passage {match.group(1)}"
                                in_passage = True
                                continue
                        
                        # Check for questions start (end of passage)
                        if re.search(r'Questions?\s+\d+', line, re.IGNORECASE):
                            in_passage = False
                            break
                        
                        # Collect passage content
                        if in_passage and len(line) > 20:  # Skip very short lines
                            passage_content.append(line)
                    
                    if passage_content and len(' '.join(passage_content)) > 300:
                        content = self.clean_text(' '.join(passage_content))
                        word_count = len(content.split())
                        
                        # Extract passage number
                        passage_num = 1
                        if passage_title:
                            match = re.search(r'(\d+)', passage_title)
                            if match:
                                passage_num = int(match.group(1))
                        
                        passages.append(CleanReadingPassage(
                            title=passage_title or f"Reading Passage {passage_num}",
                            content=content,
                            word_count=word_count,
                            difficulty="Medium",
                            topic="General",
                            passage_number=passage_num
                        ))
            
            doc.close()
            return passages
            
        except Exception as e:
            print(f"Error extracting passages from {pdf_path}: {e}")
            return []
    
    def extract_questions_clean(self, pdf_path: str) -> List[CleanReadingQuestion]:
        """Extract clean questions"""
        try:
            doc = fitz.open(pdf_path)
            questions = []
            question_id = 1
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Look for question patterns
                lines = text.split('\n')
                current_question = None
                current_passage = 1
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for passage marker
                    if re.search(r'READING\s*PASSAGE\s*(\d+)', line, re.IGNORECASE):
                        match = re.search(r'READING\s*PASSAGE\s*(\d+)', line, re.IGNORECASE)
                        if match:
                            current_passage = int(match.group(1))
                        continue
                    
                    # Check for question start
                    if re.search(r'Questions?\s+(\d+)\s*[-–]\s*(\d+)', line, re.IGNORECASE):
                        # Multiple questions range
                        match = re.search(r'Questions?\s+(\d+)\s*[-–]\s*(\d+)', line, re.IGNORECASE)
                        if match:
                            start_q = int(match.group(1))
                            end_q = int(match.group(2))
                            for q_num in range(start_q, end_q + 1):
                                questions.append(CleanReadingQuestion(
                                    id=question_id,
                                    passage_id=current_passage,
                                    type="multiple_choice",
                                    question_text=f"Question {q_num}",
                                    correct_answer="",
                                    explanation=""
                                ))
                                question_id += 1
                        continue
                    
                    # Check for individual question
                    if re.search(r'^(\d+)\.\s+(.+)', line):
                        match = re.search(r'^(\d+)\.\s+(.+)', line)
                        if match:
                            questions.append(CleanReadingQuestion(
                                id=question_id,
                                passage_id=current_passage,
                                type="multiple_choice",
                                question_text=match.group(2),
                                correct_answer="",
                                explanation=""
                            ))
                            question_id += 1
            
            doc.close()
            return questions
            
        except Exception as e:
            print(f"Error extracting questions from {pdf_path}: {e}")
            return []
    
    def parse_cambridge_test_clean(self, pdf_path: str) -> Dict[str, Any]:
        """Parse Cambridge test with clean extraction"""
        print(f"Parsing: {Path(pdf_path).name}")
        
        passages = self.extract_reading_passages_clean(pdf_path)
        questions = self.extract_questions_clean(pdf_path)
        
        return {
            "test_id": Path(pdf_path).stem,
            "title": f"Cambridge IELTS {Path(pdf_path).stem}",
            "passages": passages,
            "questions": questions,
            "total_passages": len(passages),
            "total_questions": len(questions)
        }
    
    def get_sample_content(self, pdf_path: str) -> Dict[str, Any]:
        """Get sample content from a PDF for testing"""
        result = self.parse_cambridge_test_clean(pdf_path)
        
        # Add sample content preview
        if result["passages"]:
            sample_passage = result["passages"][0]
            result["sample_passage"] = {
                "title": sample_passage.title,
                "content_preview": sample_passage.content[:500] + "..." if len(sample_passage.content) > 500 else sample_passage.content,
                "word_count": sample_passage.word_count
            }
        
        if result["questions"]:
            result["sample_questions"] = result["questions"][:5]  # First 5 questions
        
        return result

def main():
    """Test the refined parser"""
    parser = RefinedIELTSParser()
    
    # Test with Cambridge IELTS 4
    test_pdf = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/Academic/Cambridge IELTS 4 with Answers Academic [www.luckyielts.com]/Cambridge IELTS 4 with Answers Academic [www.luckyielts.com].pdf"
    
    result = parser.get_sample_content(test_pdf)
    
    print(f"📚 Test: {result['title']}")
    print(f"📄 Total Passages: {result['total_passages']}")
    print(f"❓ Total Questions: {result['total_questions']}")
    
    if "sample_passage" in result:
        sample = result["sample_passage"]
        print(f"\n📖 Sample Passage: {sample['title']}")
        print(f"Word Count: {sample['word_count']}")
        print(f"Content Preview:\n{sample['content_preview']}")
    
    if "sample_questions" in result:
        print(f"\n❓ Sample Questions:")
        for q in result["sample_questions"][:3]:
            print(f"  Q{q.id} (Passage {q.passage_id}): {q.question_text[:100]}...")

if __name__ == "__main__":
    main()
