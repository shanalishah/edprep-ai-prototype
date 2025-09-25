"""
Simple Reading Test Extractor
Extracts one clean reading passage and creates a test
"""

import fitz
import json
import re
from pathlib import Path

def extract_stepwells_test():
    """Extract the stepwells reading test"""
    
    # Extract text from PDF
    doc = fitz.open('/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Cambridge IELTS/General/Cambridge IELTS 10 with Answers GT [www.luckyielts.com].pdf')
    text = ''
    for page in doc:
        text += page.get_text() + '\n'
    doc.close()
    
    # Find reading section
    lines = text.split('\n')
    reading_start = -1
    for i, line in enumerate(lines):
        if 'READING' in line.upper() and 'PASSAGE' in line.upper():
            reading_start = i
            break
    
    if reading_start == -1:
        return None
    
    # Extract passage content
    passage_lines = []
    in_passage = False
    passage_title = ""
    
    for i in range(reading_start, min(reading_start + 100, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        
        # Check for passage title
        if 'Stepwells' in line and len(line) < 50:
            passage_title = line
            in_passage = True
            continue
        
        # Check for questions start
        if re.search(r'Questions?\s+\d+', line, re.IGNORECASE):
            break
        
        # Collect passage content
        if in_passage and len(line) > 20:
            passage_lines.append(line)
    
    if not passage_lines:
        return None
    
    # Clean and join passage
    passage_content = ' '.join(passage_lines)
    passage_content = re.sub(r'\s+', ' ', passage_content)  # Clean whitespace
    
    # Create test data
    test_data = {
        "id": 1,
        "title": "Cambridge IELTS 10 - Stepwells Reading Test",
        "time_limit": 60,
        "total_questions": 5,
        "difficulty": "medium",
        "passages": [
            {
                "id": 1,
                "title": "Stepwells",
                "content": passage_content,
                "word_count": len(passage_content.split()),
                "difficulty_level": "medium",
                "topic": "History and Culture"
            }
        ],
        "questions": [
            {
                "id": 1,
                "passage_id": 1,
                "type": "multiple_choice",
                "question_text": "What was the primary purpose of stepwells in ancient India?",
                "options": [
                    "A) Religious ceremonies",
                    "B) Access to groundwater during dry seasons", 
                    "C) Architectural decoration",
                    "D) Social gatherings"
                ],
                "correct_answer": "B",
                "explanation": "The passage states that stepwells were developed to gain access to clean, fresh groundwater during the dry season.",
                "passage_reference": "Paragraph 1"
            },
            {
                "id": 2,
                "passage_id": 1,
                "type": "true_false_not_given",
                "question_text": "Stepwells were only used by the highest classes of society.",
                "correct_answer": "False",
                "explanation": "The passage states that stepwells were places of gathering for villagers of all but the lowest classes, meaning they were used by most people, not just the highest classes.",
                "passage_reference": "Paragraph 2"
            },
            {
                "id": 3,
                "passage_id": 1,
                "type": "sentence_completion",
                "question_text": "Stepwells are unique to the region of __________ and __________.",
                "correct_answer": "Gujarat, Rajasthan",
                "explanation": "The passage mentions that stepwells are found in Gujarat and Rajasthan.",
                "passage_reference": "Paragraph 2"
            },
            {
                "id": 4,
                "passage_id": 1,
                "type": "short_answer",
                "question_text": "What are stepwells called in Gujarat?",
                "correct_answer": "vav",
                "explanation": "The passage states that in Gujarat they are called vav.",
                "passage_reference": "Paragraph 2"
            },
            {
                "id": 5,
                "passage_id": 1,
                "type": "short_answer", 
                "question_text": "What are stepwells called in Rajasthan?",
                "correct_answer": "baori",
                "explanation": "The passage states that in Rajasthan they are called baori.",
                "passage_reference": "Paragraph 2"
            }
        ]
    }
    
    return test_data

def main():
    """Extract and save the stepwells test"""
    print("🔄 Extracting Stepwells reading test...")
    
    test_data = extract_stepwells_test()
    
    if test_data:
        # Save to JSON
        output_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/backend/stepwells_reading_test.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully extracted reading test!")
        print(f"📚 Title: {test_data['title']}")
        print(f"📄 Passage: {test_data['passages'][0]['title']} ({test_data['passages'][0]['word_count']} words)")
        print(f"❓ Questions: {len(test_data['questions'])}")
        print(f"💾 Saved to: {output_path}")
        
        # Show sample content
        passage = test_data['passages'][0]
        print(f"\n📖 Sample Passage Content:")
        print(f"{passage['content'][:300]}...")
        
        print(f"\n❓ Sample Questions:")
        for q in test_data['questions'][:2]:
            print(f"  Q{q['id']}: {q['question_text']}")
            if q['options']:
                for opt in q['options']:
                    print(f"    {opt}")
            print(f"    Answer: {q['correct_answer']}")
            print()
    else:
        print("❌ Failed to extract reading test")

if __name__ == "__main__":
    main()
