"""
IELTS Listening Test Data Models
Defines the structure for listening tests, audio files, questions, and answers
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class ListeningQuestionType(Enum):
    """Types of listening questions"""
    FORM_COMPLETION = "form_completion"
    TABLE_COMPLETION = "table_completion"
    NOTE_COMPLETION = "note_completion"
    MULTIPLE_CHOICE = "multiple_choice"
    MATCHING = "matching"
    SENTENCE_COMPLETION = "sentence_completion"
    SHORT_ANSWER = "short_answer"

class ListeningSection(Enum):
    """IELTS Listening sections"""
    SECTION_1 = "section_1"  # Social context (conversation)
    SECTION_2 = "section_2"  # Social context (monologue)
    SECTION_3 = "section_3"  # Educational context (conversation)
    SECTION_4 = "section_4"  # Academic context (monologue)

@dataclass
class AudioTrack:
    """Represents an audio track for a listening section"""
    id: int
    section: ListeningSection
    track_number: int  # Track 1, 2, 3, 4
    file_path: str
    duration_seconds: int
    description: str

@dataclass
class ListeningQuestion:
    """Represents a listening question"""
    id: int
    section: ListeningSection
    question_number: int
    type: ListeningQuestionType
    question_text: str
    correct_answer: str
    alternative_answers: List[str] = None  # Alternative acceptable answers
    word_limit: Optional[str] = None  # "ONE WORD ONLY", "NO MORE THAN TWO WORDS", etc.
    options: List[str] = None  # For multiple choice questions
    context: str = None  # Additional context or instructions
    explanation: str = None  # Explanation for the answer

@dataclass
class ListeningTest:
    """Represents a complete IELTS listening test"""
    id: int
    title: str
    test_number: int  # Test 1, 2, 3, 4
    time_limit: int  # in minutes (30 minutes + 10 minutes transfer)
    total_questions: int  # Always 40 for IELTS
    difficulty: str
    audio_tracks: List[AudioTrack]
    questions: List[ListeningQuestion]
    instructions: str = "You will hear a number of different recordings and you will have to answer questions on what you hear. There will be time for you to read the instructions and questions and you will have a chance to check your work. All the recordings will be played once only. The test is in four sections. At the end of the test you will be given 10 minutes to transfer your answers to an answer sheet."

# Sample Listening Test Data
SAMPLE_LISTENING_TESTS = [
    ListeningTest(
        id=1,
        title="Cambridge IELTS 10 - Test 1 Listening",
        test_number=1,
        time_limit=40,  # 30 minutes listening + 10 minutes transfer
        total_questions=40,
        difficulty="Medium",
        audio_tracks=[
            AudioTrack(
                id=1,
                section=ListeningSection.SECTION_1,
                track_number=1,
                file_path="/audio/test1/01 Track 1.mp3",
                duration_seconds=300,  # 5 minutes
                description="Section 1: Conversation about holiday bookings"
            ),
            AudioTrack(
                id=2,
                section=ListeningSection.SECTION_2,
                track_number=2,
                file_path="/audio/test1/02 Track 2.mp3",
                duration_seconds=300,
                description="Section 2: Monologue about leisure club"
            ),
            AudioTrack(
                id=3,
                section=ListeningSection.SECTION_3,
                track_number=3,
                file_path="/audio/test1/03 Track 3.mp3",
                duration_seconds=300,
                description="Section 3: Conversation between student and professor"
            ),
            AudioTrack(
                id=4,
                section=ListeningSection.SECTION_4,
                track_number=4,
                file_path="/audio/test1/04 Track 4.mp3",
                duration_seconds=300,
                description="Section 4: Academic lecture about management"
            )
        ],
        questions=[
            # Section 1 Questions (1-10)
            ListeningQuestion(
                id=1,
                section=ListeningSection.SECTION_1,
                question_number=1,
                type=ListeningQuestionType.FORM_COMPLETION,
                question_text="Complete the notes below. Write ONE WORD AND/OR A NUMBER for each answer.",
                correct_answer="7",
                word_limit="ONE WORD AND/OR A NUMBER",
                context="Trip One: 12 days, 7 ...............km, £525"
            ),
            ListeningQuestion(
                id=2,
                section=ListeningSection.SECTION_1,
                question_number=2,
                type=ListeningQuestionType.FORM_COMPLETION,
                question_text="Complete the notes below. Write ONE WORD AND/OR A NUMBER for each answer.",
                correct_answer="8",
                word_limit="ONE WORD AND/OR A NUMBER",
                context="Includes: accommodation, car, one 8 ..............."
            ),
            # Section 2 Questions (11-20)
            ListeningQuestion(
                id=11,
                section=ListeningSection.SECTION_2,
                question_number=11,
                type=ListeningQuestionType.NOTE_COMPLETION,
                question_text="Write NO MORE THAN TWO WORDS for each answer.",
                correct_answer="health problems",
                alternative_answers=["health issues", "medical problems"],
                word_limit="NO MORE THAN TWO WORDS",
                context="New members should describe any 1 3 ..........................................................................."
            ),
            ListeningQuestion(
                id=12,
                section=ListeningSection.SECTION_2,
                question_number=12,
                type=ListeningQuestionType.NOTE_COMPLETION,
                question_text="Write NO MORE THAN TWO WORDS for each answer.",
                correct_answer="safety rules",
                word_limit="NO MORE THAN TWO WORDS",
                context="The 14 ................................will be explained to you before you use the equipment."
            ),
            # Section 3 Questions (21-30)
            ListeningQuestion(
                id=21,
                section=ListeningSection.SECTION_3,
                question_number=21,
                type=ListeningQuestionType.NOTE_COMPLETION,
                question_text="Write ONE WORD ONLY for each answer.",
                correct_answer="presentation",
                word_limit="ONE WORD ONLY",
                context="John needs help preparing for his 2 6 .................."
            ),
            ListeningQuestion(
                id=22,
                section=ListeningSection.SECTION_3,
                question_number=22,
                type=ListeningQuestionType.NOTE_COMPLETION,
                question_text="Write ONE WORD ONLY for each answer.",
                correct_answer="model",
                word_limit="ONE WORD ONLY",
                context="The professor advises John to make a 2 7 .......... ....................of his design."
            ),
            # Section 4 Questions (31-40)
            ListeningQuestion(
                id=31,
                section=ListeningSection.SECTION_4,
                question_number=31,
                type=ListeningQuestionType.NOTE_COMPLETION,
                question_text="Write ONE WORD ONLY for each answer.",
                correct_answer="competition",
                word_limit="ONE WORD ONLY",
                context="Business markets: greater 3 1 ...............................among companies"
            ),
            ListeningQuestion(
                id=32,
                section=ListeningSection.SECTION_4,
                question_number=32,
                type=ListeningQuestionType.NOTE_COMPLETION,
                question_text="Write ONE WORD ONLY for each answer.",
                correct_answer="global",
                word_limit="ONE WORD ONLY",
                context="increase in power of large 3 2 ..................................companies"
            )
        ]
    )
]

def load_cambridge_test():
    """Load the real Cambridge IELTS 10 Test 1 listening test"""
    import json
    import os
    
    # Load the corrected Cambridge test
    test_file = os.path.join(os.path.dirname(__file__), "..", "..", "cambridge_ielts_10_test_1_listening_corrected.json")
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to our system format
        audio_tracks = []
        for track_data in data["audio_tracks"]:
            track = AudioTrack(
                id=track_data["id"],
                section=ListeningSection(track_data["section"]),
                track_number=track_data["track_number"],
                file_path=track_data["file_path"],
                duration_seconds=track_data["duration_seconds"],
                description=track_data["description"]
            )
            audio_tracks.append(track)
        
        questions = []
        for question_data in data["questions"]:
            # Convert question type string to enum
            question_type = ListeningQuestionType.NOTE_COMPLETION  # Default
            if question_data["type"] == "form_completion":
                question_type = ListeningQuestionType.FORM_COMPLETION
            elif question_data["type"] == "table_completion":
                question_type = ListeningQuestionType.TABLE_COMPLETION
            elif question_data["type"] == "multiple_choice":
                question_type = ListeningQuestionType.MULTIPLE_CHOICE
            elif question_data["type"] == "matching":
                question_type = ListeningQuestionType.MATCHING
            elif question_data["type"] == "sentence_completion":
                question_type = ListeningQuestionType.SENTENCE_COMPLETION
            elif question_data["type"] == "short_answer":
                question_type = ListeningQuestionType.SHORT_ANSWER
            
            # Convert section string to enum
            section = ListeningSection.SECTION_1  # Default
            if question_data["section"] == "section_2":
                section = ListeningSection.SECTION_2
            elif question_data["section"] == "section_3":
                section = ListeningSection.SECTION_3
            elif question_data["section"] == "section_4":
                section = ListeningSection.SECTION_4
            
            question = ListeningQuestion(
                id=question_data["id"],
                section=section,
                question_number=question_data["question_number"],
                type=question_type,
                question_text=question_data["question_text"],
                correct_answer=question_data.get("correct_answer", ""),
                alternative_answers=question_data.get("alternative_answers"),
                word_limit=question_data.get("word_limit"),
                options=question_data.get("options"),
                context=question_data.get("context"),
                explanation=question_data.get("explanation")
            )
            questions.append(question)
        
        # Create listening test with ID 1 to replace the old sample test
        listening_test = ListeningTest(
            id=1,  # Use ID 1 to replace the old sample test
            title=data["title"],
            test_number=data["test_number"],
            time_limit=data["time_limit"],
            total_questions=data["total_questions"],
            difficulty=data["difficulty"],
            audio_tracks=audio_tracks,
            questions=questions
        )
        
        return listening_test
    return None

def get_listening_test(test_id: int) -> Optional[ListeningTest]:
    """Get a specific listening test by ID"""
    # Check Cambridge test first (prioritize corrected version)
    cambridge_test = load_cambridge_test()
    if cambridge_test and cambridge_test.id == test_id:
        return cambridge_test
    
    # Check sample tests
    for test in SAMPLE_LISTENING_TESTS:
        if test.id == test_id:
            return test
    
    return None

def get_all_listening_tests() -> List[ListeningTest]:
    """Get all available listening tests"""
    tests = SAMPLE_LISTENING_TESTS.copy()
    
    # Add the real Cambridge test
    cambridge_test = load_cambridge_test()
    if cambridge_test:
        tests.append(cambridge_test)
    
    return tests

def get_questions_by_section(test_id: int, section: ListeningSection) -> List[ListeningQuestion]:
    """Get questions for a specific section in a test"""
    test = get_listening_test(test_id)
    if not test:
        return []
    
    return [q for q in test.questions if q.section == section]

def get_audio_track(test_id: int, section: ListeningSection) -> Optional[AudioTrack]:
    """Get audio track for a specific section"""
    test = get_listening_test(test_id)
    if not test:
        return None
    
    for track in test.audio_tracks:
        if track.section == section:
            return track
    return None
