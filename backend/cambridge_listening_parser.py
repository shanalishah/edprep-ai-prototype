"""
Cambridge IELTS Listening Test Parser
Extracts listening questions, answers, and audio information from Cambridge PDFs
"""

import fitz
import re
import json
import os
from typing import List, Dict, Any, Optional

class CambridgeListeningParser:
    def __init__(self):
        self.pdf_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/Academic/Cambridge IELTS 10 with Answers Academic [www.luckyielts.com]/Cambridge IELTS 10 with Answers Academic [www.luckyielts.com].pdf"
        self.audio_base_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/Academic/Cambridge IELTS 10 with Answers Academic [www.luckyielts.com]"
    
    def parse_listening_test(self, test_number: int = 1) -> Dict[str, Any]:
        """Parse a complete listening test from the PDF"""
        print(f"🎧 Parsing Cambridge IELTS 10 - Test {test_number} Listening...")
        
        doc = fitz.open(self.pdf_path)
        
        # Find the listening test pages
        test_pages = self._find_listening_test_pages(doc, test_number)
        print(f"📄 Found listening test on pages: {test_pages}")
        
        # Extract questions for each section
        sections = {}
        for section_num in range(1, 5):
            section_pages = self._get_section_pages(test_pages, section_num)
            if section_pages:
                section_data = self._extract_section(doc, section_pages, section_num, test_number)
                if section_data:
                    sections[f"section_{section_num}"] = section_data
        
        # Extract answer keys
        answer_keys = self._extract_answer_keys(doc, test_number)
        
        doc.close()
        
        # Create complete test structure
        test_data = {
            "id": test_number,
            "title": f"Cambridge IELTS 10 - Test {test_number} Listening",
            "test_number": test_number,
            "time_limit": 40,  # 30 minutes + 10 minutes transfer
            "total_questions": 40,
            "difficulty": "Medium",
            "audio_tracks": self._create_audio_tracks(test_number),
            "sections": sections,
            "answer_keys": answer_keys,
            "questions": self._flatten_questions(sections)
        }
        
        return test_data
    
    def _find_listening_test_pages(self, doc, test_number: int) -> List[int]:
        """Find the page numbers for a specific listening test"""
        test_pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Look for listening test indicators
            if re.search(rf'Test {test_number}.*LISTENING|LISTENING.*Test {test_number}', text, re.IGNORECASE):
                test_pages.append(page_num + 1)
        
        return test_pages
    
    def _get_section_pages(self, test_pages: List[int], section_num: int) -> List[int]:
        """Get pages for a specific section"""
        # This is a simplified approach - in reality, we'd need more sophisticated parsing
        # For now, we'll estimate based on typical Cambridge structure
        if not test_pages:
            return []
        
        start_page = test_pages[0] + (section_num - 1) * 2  # Rough estimate
        return [start_page, start_page + 1] if start_page <= max(test_pages) else []
    
    def _extract_section(self, doc, pages: List[int], section_num: int, test_number: int) -> Optional[Dict[str, Any]]:
        """Extract questions and content for a specific section"""
        section_data = {
            "section_number": section_num,
            "questions": [],
            "instructions": "",
            "context": ""
        }
        
        for page_num in pages:
            if page_num > len(doc):
                continue
                
            page = doc[page_num - 1]  # Convert to 0-based index
            text = page.get_text()
            
            # Extract questions from this page
            questions = self._extract_questions_from_text(text, section_num, test_number)
            section_data["questions"].extend(questions)
            
            # Extract instructions
            if not section_data["instructions"]:
                instructions = self._extract_instructions(text)
                if instructions:
                    section_data["instructions"] = instructions
        
        return section_data if section_data["questions"] else None
    
    def _extract_questions_from_text(self, text: str, section_num: int, test_number: int) -> List[Dict[str, Any]]:
        """Extract individual questions from text"""
        questions = []
        lines = text.split('\n')
        
        current_question = None
        question_id = (section_num - 1) * 10 + 1  # Section 1: 1-10, Section 2: 11-20, etc.
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for question patterns
            if re.match(r'^\d+\.?\s*$', line) or re.match(r'^Questions?\s+\d+', line, re.IGNORECASE):
                if current_question:
                    questions.append(current_question)
                    question_id += 1
                
                current_question = {
                    "id": question_id,
                    "section": f"section_{section_num}",
                    "question_number": question_id - (section_num - 1) * 10,
                    "type": self._determine_question_type(text),
                    "question_text": line,
                    "correct_answer": "",
                    "word_limit": self._extract_word_limit(text),
                    "context": ""
                }
            
            elif current_question and line:
                # Add to question context
                if not current_question["context"]:
                    current_question["context"] = line
                else:
                    current_question["context"] += " " + line
        
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def _determine_question_type(self, text: str) -> str:
        """Determine the type of listening question"""
        text_lower = text.lower()
        
        if "complete the form" in text_lower or "complete the notes" in text_lower:
            return "form_completion"
        elif "complete the table" in text_lower:
            return "table_completion"
        elif "choose" in text_lower and ("letter" in text_lower or "a, b, c" in text_lower):
            return "multiple_choice"
        elif "match" in text_lower:
            return "matching"
        elif "complete the sentences" in text_lower:
            return "sentence_completion"
        else:
            return "note_completion"
    
    def _extract_word_limit(self, text: str) -> Optional[str]:
        """Extract word limit instructions"""
        word_limit_patterns = [
            r'Write ONE WORD ONLY',
            r'Write NO MORE THAN TWO WORDS',
            r'Write ONE WORD AND/OR A NUMBER',
            r'Write NO MORE THAN THREE WORDS'
        ]
        
        for pattern in word_limit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_instructions(self, text: str) -> Optional[str]:
        """Extract section instructions"""
        # Look for instruction patterns
        instruction_patterns = [
            r'Complete the notes below\.',
            r'Complete the form below\.',
            r'Complete the table below\.',
            r'Choose the correct letter, A, B or C\.'
        ]
        
        for pattern in instruction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_answer_keys(self, doc, test_number: int) -> Dict[str, str]:
        """Extract answer keys for the test"""
        answer_keys = {}
        
        # Look for answer key pages (usually near the end)
        for page_num in range(len(doc) - 20, len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if "Answer Keys" in text or "Answer Key" in text:
                # Extract answers from this page
                lines = text.split('\n')
                current_question = None
                
                for line in lines:
                    line = line.strip()
                    if re.match(r'^\d+$', line):
                        current_question = int(line)
                    elif current_question and line and not line.isdigit():
                        # This is likely an answer
                        answer_keys[str(current_question)] = line
                        current_question = None
        
        return answer_keys
    
    def _create_audio_tracks(self, test_number: int) -> List[Dict[str, Any]]:
        """Create audio track information"""
        audio_tracks = []
        
        for track_num in range(1, 5):
            audio_file = f"test{test_number}/{track_num:02d} Track {track_num}.mp3"
            audio_path = os.path.join(self.audio_base_path, audio_file)
            
            if os.path.exists(audio_path):
                audio_tracks.append({
                    "id": track_num,
                    "section": f"section_{track_num}",
                    "track_number": track_num,
                    "file_path": audio_path,
                    "duration_seconds": 300,  # Estimated 5 minutes per section
                    "description": f"Section {track_num}: {'Social conversation' if track_num == 1 else 'Social monologue' if track_num == 2 else 'Educational conversation' if track_num == 3 else 'Academic lecture'}"
                })
        
        return audio_tracks
    
    def _flatten_questions(self, sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten all questions from all sections"""
        all_questions = []
        
        for section_key, section_data in sections.items():
            if "questions" in section_data:
                all_questions.extend(section_data["questions"])
        
        return all_questions
    
    def save_test_data(self, test_data: Dict[str, Any], filename: str = None) -> str:
        """Save test data to JSON file"""
        if not filename:
            filename = f"cambridge_ielts_10_test_{test_data['test_number']}_listening.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved listening test data to: {filepath}")
        return filepath

def main():
    """Main function to parse and save listening test data"""
    parser = CambridgeListeningParser()
    
    # Parse Test 1
    test_data = parser.parse_listening_test(1)
    
    # Save to file
    filepath = parser.save_test_data(test_data)
    
    print(f"\n🎧 Cambridge IELTS 10 - Test 1 Listening Parsed Successfully!")
    print(f"📊 Questions found: {len(test_data['questions'])}")
    print(f"🎵 Audio tracks: {len(test_data['audio_tracks'])}")
    print(f"🔑 Answer keys: {len(test_data['answer_keys'])}")
    
    # Display sample data
    print(f"\n📋 Sample Questions:")
    for i, question in enumerate(test_data['questions'][:5]):
        print(f"  Q{question['id']}: {question['question_text'][:50]}...")
    
    print(f"\n🎵 Audio Tracks:")
    for track in test_data['audio_tracks']:
        print(f"  Track {track['track_number']}: {track['description']}")

if __name__ == "__main__":
    main()
