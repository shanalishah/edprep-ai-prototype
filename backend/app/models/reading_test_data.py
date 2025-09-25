"""
IELTS Reading Test Data Structure
Contains sample reading tests with passages, questions, and answers
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE_NOT_GIVEN = "true_false_not_given"
    MATCHING_HEADINGS = "matching_headings"
    SENTENCE_COMPLETION = "sentence_completion"
    SUMMARY_COMPLETION = "summary_completion"
    MATCHING_FEATURES = "matching_features"
    SHORT_ANSWER = "short_answer"
    DIAGRAM_LABELLING = "diagram_labelling"

@dataclass
class Question:
    id: int
    type: QuestionType
    question_text: str
    correct_answer: str
    explanation: str
    options: List[str] = None  # For multiple choice
    passage_reference: str = None  # Which part of passage the answer is in

@dataclass
class ReadingPassage:
    id: int
    title: str
    content: str
    word_count: int
    difficulty_level: str  # Easy, Medium, Hard
    topic: str

@dataclass
class ReadingTest:
    id: int
    title: str
    passages: List[ReadingPassage]
    questions: List[Question]
    time_limit: int  # in minutes
    total_questions: int
    difficulty: str

# Load official Stepwells test
def load_official_stepwells_test():
    """Load the official Cambridge IELTS Stepwells test"""
    import json
    import os
    
    # Load the complete stepwells test
    test_file = os.path.join(os.path.dirname(__file__), "..", "..", "complete_stepwells_test.json")
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
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
    return None

# Sample Reading Test Data
SAMPLE_READING_TESTS = [
    ReadingTest(
        id=1,
        title="IELTS Academic Reading Test 1 - The Future of Work",
        time_limit=60,
        total_questions=40,
        difficulty="Medium",
        passages=[
            ReadingPassage(
                id=1,
                title="The Rise of Remote Work",
                content="""
The concept of remote work has evolved dramatically over the past decade, transforming from a rare privilege to a mainstream employment model. This shift has been accelerated by technological advancements, changing societal attitudes, and global events that have forced organizations to rethink traditional workplace structures.

Remote work, also known as telecommuting or telework, refers to a work arrangement where employees perform their job duties from locations outside the traditional office environment. This can include working from home, co-working spaces, or any other location with internet connectivity. The rise of remote work has been facilitated by several key factors.

First, technological infrastructure has improved significantly. High-speed internet, cloud computing, video conferencing platforms, and collaborative software tools have made it possible for teams to work together effectively regardless of their physical location. These technologies have eliminated many of the barriers that previously made remote work challenging.

Second, there has been a cultural shift in how work-life balance is perceived. Many employees now prioritize flexibility and autonomy over traditional office perks. This change in mindset has been particularly pronounced among younger generations, who value the ability to work from anywhere and maintain better work-life integration.

Third, employers have recognized the potential benefits of remote work arrangements. These include reduced overhead costs, access to a global talent pool, increased employee satisfaction, and improved retention rates. Many companies have found that remote work can actually increase productivity when implemented correctly.

However, remote work also presents challenges. Communication can become more complex when team members are not physically present. Building company culture and maintaining team cohesion requires more intentional effort. Some employees may struggle with isolation or find it difficult to separate work and personal life when working from home.

The future of remote work appears to be hybrid in nature. Many organizations are adopting flexible models that combine remote and in-person work, allowing employees to choose the arrangement that works best for their role and personal circumstances. This approach seeks to capture the benefits of both remote and traditional work arrangements while minimizing their respective drawbacks.

As we look ahead, it's clear that remote work will continue to evolve. New technologies, such as virtual reality and augmented reality, may further enhance remote collaboration capabilities. The key to success will be finding the right balance between flexibility and structure, technology and human connection, and individual autonomy and team collaboration.
                """,
                word_count=450,
                difficulty_level="Medium",
                topic="Work and Employment"
            ),
            ReadingPassage(
                id=2,
                title="Artificial Intelligence in the Workplace",
                content="""
Artificial Intelligence (AI) is rapidly transforming the modern workplace, creating both opportunities and challenges for workers and organizations alike. As AI technologies become more sophisticated and accessible, their impact on various industries continues to grow, fundamentally altering how work is performed and what skills are valued.

AI encompasses a broad range of technologies, including machine learning, natural language processing, computer vision, and robotics. These technologies are being integrated into workplace systems to automate routine tasks, enhance decision-making processes, and improve overall efficiency. From customer service chatbots to predictive analytics in healthcare, AI applications are becoming increasingly prevalent across diverse sectors.

One of the most significant impacts of AI on the workplace is automation. Many routine, repetitive tasks that were previously performed by humans are now being automated using AI systems. This includes data entry, basic customer service inquiries, and even some aspects of financial analysis. While this automation can lead to increased efficiency and reduced costs for organizations, it also raises concerns about job displacement.

However, the relationship between AI and employment is more complex than simple replacement. AI is also creating new types of jobs and transforming existing roles. Positions such as AI specialists, data scientists, and machine learning engineers are in high demand. Additionally, many traditional jobs are evolving to incorporate AI tools, requiring workers to develop new skills and adapt to changing work processes.

The integration of AI in the workplace also presents opportunities for enhanced decision-making. AI systems can process vast amounts of data quickly and identify patterns that might be missed by human analysis. This capability is particularly valuable in fields such as healthcare, finance, and marketing, where data-driven insights can lead to better outcomes.

Despite these benefits, the widespread adoption of AI in the workplace raises important ethical and practical considerations. Issues such as algorithmic bias, data privacy, and the need for human oversight of AI systems are becoming increasingly important. Organizations must carefully consider how to implement AI technologies in ways that are fair, transparent, and beneficial to all stakeholders.

Looking forward, the successful integration of AI in the workplace will likely depend on finding the right balance between automation and human skills. While AI can handle many routine tasks, human capabilities such as creativity, emotional intelligence, and complex problem-solving remain valuable and difficult to replicate. The future workplace may be characterized by human-AI collaboration, where each complements the other's strengths.
                """,
                word_count=420,
                difficulty_level="Medium",
                topic="Technology and AI"
            ),
            ReadingPassage(
                id=3,
                title="Sustainable Business Practices",
                content="""
Sustainability has become a central concern for businesses worldwide, driven by increasing awareness of environmental challenges, regulatory pressures, and changing consumer expectations. Companies are recognizing that sustainable practices are not just ethical imperatives but also strategic advantages that can drive innovation, reduce costs, and enhance brand reputation.

Sustainable business practices encompass a wide range of activities aimed at minimizing negative environmental and social impacts while maximizing positive contributions to society. This includes reducing carbon emissions, conserving natural resources, promoting social equity, and ensuring ethical supply chain management. The concept of sustainability extends beyond environmental concerns to include economic and social dimensions, often referred to as the "triple bottom line."

One of the most significant drivers of sustainable business practices is climate change. As the effects of global warming become more apparent, businesses are under increasing pressure to reduce their carbon footprint and transition to renewable energy sources. Many companies have set ambitious targets for carbon neutrality, investing in renewable energy infrastructure and implementing energy-efficient technologies.

Resource conservation is another key aspect of sustainable business practices. Companies are finding innovative ways to reduce waste, recycle materials, and use resources more efficiently. This includes everything from implementing circular economy principles to developing products with longer lifespans and better recyclability.

Social sustainability is equally important, focusing on fair labor practices, community engagement, and social equity. Companies are increasingly expected to ensure that their operations benefit local communities and that their supply chains are free from exploitation. This includes paying fair wages, providing safe working conditions, and supporting local economic development.

The business case for sustainability is becoming increasingly compelling. Companies that adopt sustainable practices often experience reduced operational costs through improved efficiency and waste reduction. They also benefit from enhanced brand reputation, increased customer loyalty, and improved employee satisfaction. Additionally, sustainable companies are often better positioned to attract investment and talent.

However, implementing sustainable practices can present challenges. Initial investments in sustainable technologies and processes can be significant, and the benefits may take time to materialize. Companies must also navigate complex regulatory environments and ensure that their sustainability initiatives are genuine and not merely marketing exercises.

The future of sustainable business practices will likely involve greater integration of sustainability into core business strategies. This means moving beyond isolated initiatives to embedding sustainability principles throughout all aspects of business operations. Companies that successfully integrate sustainability into their business models will be better positioned to thrive in an increasingly environmentally and socially conscious marketplace.
                """,
                word_count=430,
                difficulty_level="Medium",
                topic="Environment and Sustainability"
            )
        ],
        questions=[
            # Passage 1 Questions
            Question(
                id=1,
                type=QuestionType.MULTIPLE_CHOICE,
                question_text="According to the passage, what has been the main factor in the rise of remote work?",
                options=[
                    "A) Reduced office costs",
                    "B) Technological advancements",
                    "C) Employee preferences",
                    "D) Government regulations"
                ],
                correct_answer="B",
                explanation="The passage states that 'technological infrastructure has improved significantly' and lists various technologies that have 'eliminated many of the barriers that previously made remote work challenging.'",
                passage_reference="Paragraph 3"
            ),
            Question(
                id=2,
                type=QuestionType.TRUE_FALSE_NOT_GIVEN,
                question_text="Remote work always increases productivity.",
                correct_answer="False",
                explanation="The passage states that 'Many companies have found that remote work can actually increase productivity when implemented correctly,' which implies it doesn't always increase productivity.",
                passage_reference="Paragraph 5"
            ),
            Question(
                id=3,
                type=QuestionType.SENTENCE_COMPLETION,
                question_text="The future of remote work appears to be _____ in nature.",
                correct_answer="hybrid",
                explanation="The passage explicitly states 'The future of remote work appears to be hybrid in nature.'",
                passage_reference="Paragraph 6"
            ),
            Question(
                id=4,
                type=QuestionType.MULTIPLE_CHOICE,
                question_text="What challenge is mentioned regarding remote work?",
                options=[
                    "A) Higher costs",
                    "B) Communication complexity",
                    "C) Technology limitations",
                    "D) Legal issues"
                ],
                correct_answer="B",
                explanation="The passage states that 'Communication can become more complex when team members are not physically present.'",
                passage_reference="Paragraph 5"
            ),
            Question(
                id=5,
                type=QuestionType.TRUE_FALSE_NOT_GIVEN,
                question_text="All employees prefer remote work over traditional office work.",
                correct_answer="False",
                explanation="The passage mentions that 'Some employees may struggle with isolation or find it difficult to separate work and personal life when working from home,' indicating not all employees prefer remote work.",
                passage_reference="Paragraph 5"
            ),
            
            # Passage 2 Questions
            Question(
                id=6,
                type=QuestionType.MULTIPLE_CHOICE,
                question_text="What is the main impact of AI on the workplace according to the passage?",
                options=[
                    "A) Increased job security",
                    "B) Automation of routine tasks",
                    "C) Reduced need for human skills",
                    "D) Simplified work processes"
                ],
                correct_answer="B",
                explanation="The passage states that 'One of the most significant impacts of AI on the workplace is automation' and specifically mentions 'routine, repetitive tasks.'",
                passage_reference="Paragraph 3"
            ),
            Question(
                id=7,
                type=QuestionType.TRUE_FALSE_NOT_GIVEN,
                question_text="AI is creating new types of jobs.",
                correct_answer="True",
                explanation="The passage explicitly states 'AI is also creating new types of jobs' and gives examples like 'AI specialists, data scientists, and machine learning engineers.'",
                passage_reference="Paragraph 4"
            ),
            Question(
                id=8,
                type=QuestionType.SENTENCE_COMPLETION,
                question_text="AI systems can process vast amounts of data quickly and identify _____ that might be missed by human analysis.",
                correct_answer="patterns",
                explanation="The passage states that AI systems 'can process vast amounts of data quickly and identify patterns that might be missed by human analysis.'",
                passage_reference="Paragraph 5"
            ),
            Question(
                id=9,
                type=QuestionType.MULTIPLE_CHOICE,
                question_text="What is mentioned as a concern about AI in the workplace?",
                options=[
                    "A) High implementation costs",
                    "B) Algorithmic bias",
                    "C) Limited applications",
                    "D) Employee resistance"
                ],
                correct_answer="B",
                explanation="The passage mentions 'algorithmic bias' as one of the important ethical and practical considerations raised by AI adoption.",
                passage_reference="Paragraph 6"
            ),
            Question(
                id=10,
                type=QuestionType.TRUE_FALSE_NOT_GIVEN,
                question_text="Human skills like creativity and emotional intelligence are easily replicated by AI.",
                correct_answer="False",
                explanation="The passage states that 'human capabilities such as creativity, emotional intelligence, and complex problem-solving remain valuable and difficult to replicate.'",
                passage_reference="Paragraph 7"
            ),
            
            # Passage 3 Questions
            Question(
                id=11,
                type=QuestionType.MULTIPLE_CHOICE,
                question_text="What is the 'triple bottom line' mentioned in the passage?",
                options=[
                    "A) Environmental, economic, and social dimensions",
                    "B) Profit, people, and planet",
                    "C) Cost, quality, and time",
                    "D) Innovation, efficiency, and growth"
                ],
                correct_answer="A",
                explanation="The passage states that sustainability extends beyond environmental concerns to include economic and social dimensions, often referred to as the triple bottom line.",
                passage_reference="Paragraph 2"
            ),
            Question(
                id=12,
                type=QuestionType.TRUE_FALSE_NOT_GIVEN,
                question_text="Climate change is the only driver of sustainable business practices.",
                correct_answer="False",
                explanation="The passage states that climate change is 'One of the most significant drivers' but also mentions 'regulatory pressures and changing consumer expectations' as other drivers.",
                passage_reference="Paragraph 1 and 3"
            ),
            Question(
                id=13,
                type=QuestionType.SENTENCE_COMPLETION,
                question_text="Companies are finding innovative ways to reduce waste, recycle materials, and use resources more _____.",
                correct_answer="efficiently",
                explanation="The passage states that 'Companies are finding innovative ways to reduce waste, recycle materials, and use resources more efficiently.'",
                passage_reference="Paragraph 4"
            ),
            Question(
                id=14,
                type=QuestionType.MULTIPLE_CHOICE,
                question_text="What benefit of sustainable practices is mentioned in the passage?",
                options=[
                    "A) Immediate cost savings",
                    "B) Enhanced brand reputation",
                    "C) Simplified operations",
                    "D) Reduced competition"
                ],
                correct_answer="B",
                explanation="The passage mentions 'enhanced brand reputation' as one of the benefits of sustainable practices.",
                passage_reference="Paragraph 6"
            ),
            Question(
                id=15,
                type=QuestionType.TRUE_FALSE_NOT_GIVEN,
                question_text="Implementing sustainable practices always results in immediate financial benefits.",
                correct_answer="False",
                explanation="The passage states that 'the benefits may take time to materialize,' indicating that immediate benefits are not guaranteed.",
                passage_reference="Paragraph 7"
            )
        ]
    )
]

def get_reading_test(test_id: int) -> ReadingTest:
    """Get a specific reading test by ID"""
    # Check sample tests first
    for test in SAMPLE_READING_TESTS:
        if test.id == test_id:
            return test
    
    # Check official Stepwells test
    official_stepwells = load_official_stepwells_test()
    if official_stepwells and official_stepwells.id == test_id:
        return official_stepwells
    
    return None

def get_all_reading_tests() -> List[ReadingTest]:
    """Get all available reading tests"""
    tests = SAMPLE_READING_TESTS.copy()
    
    # Add the official Stepwells test
    official_stepwells = load_official_stepwells_test()
    if official_stepwells:
        tests.append(official_stepwells)
    
    return tests

def get_questions_by_passage(test_id: int, passage_id: int) -> List[Question]:
    """Get questions for a specific passage in a test"""
    test = get_reading_test(test_id)
    if not test:
        return []
    
    # For this sample, questions 1-5 are for passage 1, 6-10 for passage 2, 11-15 for passage 3
    if passage_id == 1:
        return [q for q in test.questions if q.id <= 5]
    elif passage_id == 2:
        return [q for q in test.questions if 6 <= q.id <= 10]
    elif passage_id == 3:
        return [q for q in test.questions if 11 <= q.id <= 15]
    
    return []
