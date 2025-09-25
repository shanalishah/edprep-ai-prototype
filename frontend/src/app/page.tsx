'use client';

import { useState } from 'react';

interface ScoringResult {
  task_achievement: number;
  coherence_cohesion: number;
  lexical_resource: number;
  grammatical_range_accuracy: number;
  overall_band_score: number;
  feedback: string;
}

export default function Home() {
  const [essay, setEssay] = useState('');
  const [prompt, setPrompt] = useState('Some people believe that technology has made our lives more complicated, while others think it has made life easier. Discuss both views and give your opinion.');
  const [result, setResult] = useState<ScoringResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!essay.trim()) {
      setError('Please write an essay before submitting.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = 'https://edprep-ai-backend.onrender.com';
      const response = await fetch(`${apiUrl}/assess`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          essay,
          task_type: 'Task 2',
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h1 className="text-4xl font-bold text-center text-blue-600 mb-8">
            🎓 EdPrep AI - IELTS Writing Assessment
          </h1>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Writing Prompt:
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={3}
                placeholder="Enter your IELTS writing prompt..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Your Essay:
              </label>
              <textarea
                value={essay}
                onChange={(e) => setEssay(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={10}
                placeholder="Write your essay here..."
              />
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Assessing Essay...' : '📊 Assess Essay'}
            </button>

            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                <strong>Error:</strong> {error}
              </div>
            )}

            {result && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                <h2 className="text-2xl font-bold text-green-800 mb-4">Assessment Results</h2>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white p-4 rounded-lg shadow">
                    <div className="text-sm text-gray-600">Task Achievement</div>
                    <div className="text-2xl font-bold text-blue-600">{result.task_achievement.toFixed(1)}</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow">
                    <div className="text-sm text-gray-600">Coherence & Cohesion</div>
                    <div className="text-2xl font-bold text-blue-600">{result.coherence_cohesion.toFixed(1)}</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow">
                    <div className="text-sm text-gray-600">Lexical Resource</div>
                    <div className="text-2xl font-bold text-blue-600">{result.lexical_resource.toFixed(1)}</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg shadow">
                    <div className="text-sm text-gray-600">Grammar Range</div>
                    <div className="text-2xl font-bold text-blue-600">{result.grammatical_range_accuracy.toFixed(1)}</div>
                  </div>
                </div>

                <div className="bg-blue-100 p-4 rounded-lg mb-4">
                  <div className="text-lg font-semibold text-blue-800">Overall Band Score</div>
                  <div className="text-3xl font-bold text-blue-600">{result.overall_band_score.toFixed(1)}</div>
                </div>

                <div className="bg-yellow-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-gray-800 mb-2">Feedback:</h3>
                  <div className="text-gray-700 whitespace-pre-wrap">{result.feedback}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}