"""
Improved Cambridge IELTS Listening Test Parser
Extracts listening questions and answers from specific pages
"""

import fitz
import re
import json
import os
from typing import List, Dict, Any, Optional

class ImprovedListeningParser:
    def __init__(self):
        self.pdf_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/Academic/Cambridge IELTS 10 with Answers Academic [www.luckyielts.com]/Cambridge IELTS 10 with Answers Academic [www.luckyielts.com].pdf"
        self.audio_base_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/Academic/Cambridge IELTS 10 with Answers Academic [www.luckyielts.com]"
        
        # Define listening test page ranges based on our analysis
        self.test_pages = {
            1: [3, 4, 5, 6, 7, 8, 9],      # Test 1: pages 3-9
            2: [26, 27, 28, 29, 30, 31, 32, 33],  # Test 2: pages 26-33
            3: [50, 51, 52, 53, 54, 55, 56],      # Test 3: pages 50-56
            4: [73, 74, 75, 76, 77, 78, 79, 80]   # Test 4: pages 73-80
        }
    
    def parse_listening_test(self, test_number: int = 1) -> Dict[str, Any]:
        """Parse a complete listening test from the PDF"""
        print(f"🎧 Parsing Cambridge IELTS 10 - Test {test_number} Listening...")
        
        doc = fitz.open(self.pdf_path)
        
        # Get pages for this test
        pages = self.test_pages.get(test_number, [])
        print(f"📄 Processing pages: {pages}")
        
        # Extract content from all pages
        all_text = ""
        for page_num in pages:
            if page_num <= len(doc):
                page = doc[page_num - 1]  # Convert to 0-based index
                all_text += page.get_text() + "\n"
        
        # Parse the content
        questions = self._parse_questions_from_text(all_text, test_number)
        
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
            "questions": questions,
            "answer_keys": answer_keys
        }
        
        return test_data
    
    def _parse_questions_from_text(self, text: str, test_number: int) -> List[Dict[str, Any]]:
        """Parse questions from the combined text"""
        questions = []
        lines = text.split('\n')
        
        current_question = None
        question_id = 1
        current_section = 1
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Detect section changes
            if re.search(r'SECTION \d+', line, re.IGNORECASE):
                section_match = re.search(r'SECTION (\d+)', line, re.IGNORECASE)
                if section_match:
                    current_section = int(section_match.group(1))
                    question_id = (current_section - 1) * 10 + 1
                continue
            
            # Look for question patterns
            if (re.match(r'^\d+\.?\s*$', line) or 
                re.match(r'^Questions?\s+\d+', line, re.IGNORECASE) or
                re.match(r'^\d+\s*$', line)):
                
                if current_question:
                    questions.append(current_question)
                    question_id += 1
                
                # Determine question type from context
                question_type = self._determine_question_type(text, i)
                word_limit = self._extract_word_limit_from_context(text, i)
                
                current_question = {
                    "id": question_id,
                    "section": f"section_{current_section}",
                    "question_number": question_id - (current_section - 1) * 10,
                    "type": question_type,
                    "question_text": line,
                    "correct_answer": "",
                    "word_limit": word_limit,
                    "context": "",
                    "options": []
                }
            
            elif current_question and line:
                # Add to question context or options
                if self._is_option_line(line):
                    current_question["options"].append(line)
                else:
                    if not current_question["context"]:
                        current_question["context"] = line
                    else:
                        current_question["context"] += " " + line
        
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def _determine_question_type(self, text: str, line_index: int) -> str:
        """Determine question type from context"""
        # Look at surrounding lines for context
        lines = text.split('\n')
        context_lines = lines[max(0, line_index-5):line_index+5]
        context = ' '.join(context_lines).lower()
        
        if "complete the form" in context:
            return "form_completion"
        elif "complete the table" in context:
            return "table_completion"
        elif "complete the notes" in context:
            return "note_completion"
        elif "choose" in context and ("letter" in context or "a, b, c" in context):
            return "multiple_choice"
        elif "match" in context:
            return "matching"
        elif "complete the sentences" in context:
            return "sentence_completion"
        else:
            return "note_completion"  # Default
    
    def _extract_word_limit_from_context(self, text: str, line_index: int) -> Optional[str]:
        """Extract word limit from context"""
        lines = text.split('\n')
        context_lines = lines[max(0, line_index-3):line_index+3]
        context = ' '.join(context_lines)
        
        word_limit_patterns = [
            r'Write ONE WORD ONLY',
            r'Write NO MORE THAN TWO WORDS',
            r'Write ONE WORD AND/OR A NUMBER',
            r'Write NO MORE THAN THREE WORDS'
        ]
        
        for pattern in word_limit_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _is_option_line(self, line: str) -> bool:
        """Check if a line is an option (A, B, C, D, etc.)"""
        return re.match(r'^[A-G]\s+', line) is not None
    
    def _extract_answer_keys(self, doc, test_number: int) -> Dict[str, str]:
        """Extract answer keys for the test"""
        answer_keys = {}
        
        # Answer keys are on pages 144-153
        for page_num in range(144, 154):
            if page_num <= len(doc):
                page = doc[page_num - 1]
                text = page.get_text()
                
                if "Answer Keys" in text:
                    # Extract answers from this page
                    lines = text.split('\n')
                    current_question = None
                    
                    for line in lines:
                        line = line.strip()
                        if re.match(r'^\d+$', line):
                            current_question = int(line)
                        elif current_question and line and not line.isdigit() and len(line) < 50:
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
    parser = ImprovedListeningParser()
    
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
    for i, question in enumerate(test_data['questions'][:10]):
        print(f"  Q{question['id']} ({question['section']}): {question['question_text'][:50]}...")
        if question['context']:
            print(f"    Context: {question['context'][:80]}...")
    
    print(f"\n🎵 Audio Tracks:")
    for track in test_data['audio_tracks']:
        print(f"  Track {track['track_number']}: {track['description']}")
    
    print(f"\n🔑 Sample Answer Keys:")
    for i, (q_id, answer) in enumerate(list(test_data['answer_keys'].items())[:10]):
        print(f"  Q{q_id}: {answer}")

if __name__ == "__main__":
    main()
