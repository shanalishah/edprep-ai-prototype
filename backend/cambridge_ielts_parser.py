"""
Cambridge IELTS PDF Parser
Extracts reading passages, questions, and answers from Cambridge IELTS PDFs
"""

import fitz  # pymupdf
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ReadingPassage:
    title: str
    content: str
    word_count: int
    difficulty: str = "Medium"
    topic: str = "General"

@dataclass
class ReadingQuestion:
    id: int
    type: str
    question_text: str
    options: List[str] = None
    correct_answer: str = ""
    explanation: str = ""

@dataclass
class ReadingTest:
    test_id: str
    title: str
    passages: List[ReadingPassage]
    questions: List[ReadingQuestion]
    answers: Dict[int, str]

class CambridgeIELTSParser:
    """Parser for Cambridge IELTS PDF files"""
    
    def __init__(self):
        self.cambridge_path = Path("/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def find_reading_section(self, text: str) -> str:
        """Find the reading section in the text"""
        # Look for reading section markers
        reading_patterns = [
            r"READING\s*PASSAGE\s*1",
            r"Test\s+\d+\s+READING",
            r"READING\s*SECTION",
            r"Questions\s+\d+\s*-\s*\d+"
        ]
        
        lines = text.split('\n')
        reading_start = -1
        
        for i, line in enumerate(lines):
            for pattern in reading_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    reading_start = i
                    break
            if reading_start != -1:
                break
        
        if reading_start == -1:
            return text  # Return full text if no clear reading section found
        
        # Extract from reading start to end of test
        reading_text = '\n'.join(lines[reading_start:])
        return reading_text
    
    def extract_passages(self, text: str) -> List[ReadingPassage]:
        """Extract reading passages from text"""
        passages = []
        
        # Look for passage markers
        passage_patterns = [
            r"READING\s*PASSAGE\s*(\d+)",
            r"PASSAGE\s*(\d+)",
            r"Section\s*(\d+)"
        ]
        
        # Split text into sections
        sections = re.split(r'READING\s*PASSAGE\s*\d+', text, flags=re.IGNORECASE)
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            if len(section.strip()) < 100:  # Skip very short sections
                continue
            
            # Extract title (first line or first few words)
            lines = section.strip().split('\n')
            title = f"Reading Passage {i}"
            
            # Find actual passage content (skip questions)
            content_lines = []
            in_questions = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if we've reached questions
                if re.search(r'Questions?\s+\d+', line, re.IGNORECASE):
                    in_questions = True
                    break
                
                if not in_questions and len(line) > 10:  # Skip very short lines
                    content_lines.append(line)
            
            content = ' '.join(content_lines)
            
            if len(content) > 200:  # Only include substantial passages
                word_count = len(content.split())
                passages.append(ReadingPassage(
                    title=title,
                    content=content,
                    word_count=word_count,
                    difficulty="Medium",
                    topic="General"
                ))
        
        return passages
    
    def extract_questions(self, text: str) -> List[ReadingQuestion]:
        """Extract questions from text"""
        questions = []
        
        # Look for question patterns
        question_patterns = [
            r"Questions?\s+(\d+)\s*[-–]\s*(\d+)",
            r"Questions?\s+(\d+)",
            r"(\d+)\.\s+(.+)"
        ]
        
        lines = text.split('\n')
        current_question = None
        question_id = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for question start
            for pattern in question_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if current_question:
                        questions.append(current_question)
                    
                    current_question = ReadingQuestion(
                        id=question_id,
                        type="multiple_choice",  # Default type
                        question_text=line,
                        options=[],
                        correct_answer="",
                        explanation=""
                    )
                    question_id += 1
                    break
        
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def parse_cambridge_pdf(self, pdf_path: str) -> ReadingTest:
        """Parse a Cambridge IELTS PDF file"""
        print(f"Parsing: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        # Find reading section
        reading_text = self.find_reading_section(text)
        
        # Extract passages and questions
        passages = self.extract_passages(reading_text)
        questions = self.extract_questions(reading_text)
        
        # Create test object
        test_id = Path(pdf_path).stem
        test = ReadingTest(
            test_id=test_id,
            title=f"Cambridge IELTS {test_id}",
            passages=passages,
            questions=questions,
            answers={}
        )
        
        return test
    
    def get_available_pdfs(self) -> List[str]:
        """Get list of available Cambridge IELTS PDF files"""
        pdf_files = []
        
        # Academic PDFs
        academic_path = self.cambridge_path / "Academic"
        if academic_path.exists():
            pdf_files.extend(list(academic_path.rglob("*.pdf")))
        
        # General PDFs
        general_path = self.cambridge_path / "General"
        if general_path.exists():
            pdf_files.extend(list(general_path.rglob("*.pdf")))
        
        return [str(pdf) for pdf in pdf_files]
    
    def parse_all_tests(self) -> List[ReadingTest]:
        """Parse all available Cambridge IELTS tests"""
        pdf_files = self.get_available_pdfs()
        tests = []
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files[:3]:  # Parse first 3 files for testing
            try:
                test = self.parse_cambridge_pdf(pdf_file)
                if test and test.passages:
                    tests.append(test)
                    print(f"✅ Parsed {test.title}: {len(test.passages)} passages, {len(test.questions)} questions")
                else:
                    print(f"❌ No content extracted from {pdf_file}")
            except Exception as e:
                print(f"❌ Error parsing {pdf_file}: {e}")
        
        return tests

def main():
    """Test the parser"""
    parser = CambridgeIELTSParser()
    
    # Test with one file first
    pdf_files = parser.get_available_pdfs()
    if pdf_files:
        test = parser.parse_cambridge_pdf(pdf_files[0])
        if test:
            print(f"\n📚 Test: {test.title}")
            print(f"📄 Passages: {len(test.passages)}")
            for i, passage in enumerate(test.passages, 1):
                print(f"  Passage {i}: {passage.title} ({passage.word_count} words)")
                print(f"    Content preview: {passage.content[:100]}...")
            print(f"❓ Questions: {len(test.questions)}")
        else:
            print("❌ Failed to parse test")
    else:
        print("❌ No PDF files found")

if __name__ == "__main__":
    main()
