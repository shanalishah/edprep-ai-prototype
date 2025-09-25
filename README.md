# EdPrep AI - IELTS Writing Assessment Prototype

An AI-powered IELTS writing assessment tool that provides instant feedback and band scores for IELTS essays.

## Features

- **Instant Assessment**: Get immediate feedback on IELTS essays with detailed band scores
- **Four Criteria Scoring**: Task Achievement, Coherence & Cohesion, Lexical Resource, Grammatical Range
- **Detailed Feedback**: Comprehensive feedback with specific improvement suggestions
- **Sample Prompts**: Built-in sample IELTS writing prompts for practice
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Transformers**: Hugging Face transformers for NLP models
- **OpenAI API**: Optional integration for enhanced feedback generation
- **Pydantic**: Data validation and serialization

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Responsive Design**: Mobile-friendly interface

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your configuration
```

5. Start the backend server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### POST /assess
Assess an IELTS essay and provide detailed feedback.

**Request Body:**
```json
{
  "prompt": "Your IELTS writing prompt",
  "essay": "Your essay text",
  "task_type": "Task 2"
}
```

**Response:**
```json
{
  "task_achievement": 6.5,
  "coherence_cohesion": 7.0,
  "lexical_resource": 6.0,
  "grammatical_range": 6.5,
  "overall_band_score": 6.5,
  "feedback": "Detailed feedback text...",
  "suggestions": ["Suggestion 1", "Suggestion 2"]
}
```

### GET /sample-prompts
Get sample IELTS writing prompts.

### GET /health
Health check endpoint.

## Scoring Criteria

The system evaluates essays based on the official IELTS writing criteria:

1. **Task Achievement (25%)**: How well the essay addresses the task requirements
2. **Coherence and Cohesion (25%)**: Organization and logical flow
3. **Lexical Resource (25%)**: Vocabulary range and accuracy
4. **Grammatical Range and Accuracy (25%)**: Grammar variety and correctness

## Development

### Project Structure
```
edprep-ai-prototype/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   └── models/
│   │       ├── essay_scorer.py  # Essay scoring logic
│   │       └── feedback_generator.py  # Feedback generation
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   └── page.tsx         # Main page
│   │   └── components/
│   │       ├── EssayForm.tsx    # Essay submission form
│   │       ├── ResultsDisplay.tsx  # Results display
│   │       └── SamplePrompts.tsx   # Sample prompts
│   └── package.json
└── README.md
```

### Adding New Features

1. **Backend**: Add new endpoints in `app/main.py`
2. **Frontend**: Create new components in `src/components/`
3. **Models**: Enhance scoring logic in `app/models/`

## Future Enhancements

- [ ] User authentication and progress tracking
- [ ] Advanced ML models for more accurate scoring
- [ ] Speaking practice with AI bot
- [ ] Reading and listening practice modules
- [ ] Personalized study plans
- [ ] Community features and peer review

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or support, please open an issue in the repository.

