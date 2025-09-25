'use client';

import { useState } from 'react';

export default function Home() {
  const [activeTab, setActiveTab] = useState('writing');
  const [essay, setEssay] = useState('');
  const [prompt, setPrompt] = useState('Some people believe that technology has made our lives more complicated, while others think it has made life easier. Discuss both views and give your opinion.');
  const [result, setResult] = useState<any>(null);
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
      const response = await fetch('https://edprep-ai-backend.onrender.com/assess', {
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
      <div className="max-w-6xl mx-auto px-4">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h1 className="text-4xl font-bold text-center text-blue-600 mb-8">
            🎓 EdPrep AI - Complete IELTS Assessment Platform
          </h1>

          {/* Navigation Tabs */}
          <div className="flex space-x-1 mb-8 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('writing')}
              className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${
                activeTab === 'writing'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              ✍️ Writing Assessment
            </button>
            <button
              onClick={() => setActiveTab('reading')}
              className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${
                activeTab === 'reading'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              📖 Reading Tests
            </button>
            <button
              onClick={() => setActiveTab('listening')}
              className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${
                activeTab === 'listening'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              🎧 Listening Tests
            </button>
          </div>

          {/* Writing Assessment Tab */}
          {activeTab === 'writing' && (
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
                      <div className="text-2xl font-bold text-blue-600">{result.task_achievement?.toFixed(1) || 'N/A'}</div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600">Coherence & Cohesion</div>
                      <div className="text-2xl font-bold text-blue-600">{result.coherence_cohesion?.toFixed(1) || 'N/A'}</div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600">Lexical Resource</div>
                      <div className="text-2xl font-bold text-blue-600">{result.lexical_resource?.toFixed(1) || 'N/A'}</div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600">Grammar Range</div>
                      <div className="text-2xl font-bold text-blue-600">{result.grammatical_range_accuracy?.toFixed(1) || result.grammatical_range?.toFixed(1) || 'N/A'}</div>
                    </div>
                  </div>

                  <div className="bg-blue-100 p-4 rounded-lg mb-4">
                    <div className="text-lg font-semibold text-blue-800">Overall Band Score</div>
                    <div className="text-3xl font-bold text-blue-600">{result.overall_band_score?.toFixed(1) || 'N/A'}</div>
                  </div>

                  <div className="bg-yellow-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-gray-800 mb-2">Feedback:</h3>
                    <div className="text-gray-700 whitespace-pre-wrap">{result.feedback || 'No feedback available'}</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Reading Tests Tab */}
          {activeTab === 'reading' && (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📖</div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Reading Tests</h2>
              <p className="text-gray-600 mb-6">
                Comprehensive IELTS Reading tests with multiple passages and question types.
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                <h3 className="font-semibold text-blue-800 mb-2">Available Features:</h3>
                <ul className="text-blue-700 text-left space-y-1">
                  <li>• Multiple choice questions</li>
                  <li>• True/False/Not Given</li>
                  <li>• Matching headings</li>
                  <li>• Sentence completion</li>
                  <li>• Summary completion</li>
                  <li>• Instant scoring and feedback</li>
                </ul>
              </div>
              <button className="mt-6 bg-blue-600 text-white py-2 px-6 rounded-md hover:bg-blue-700 transition-colors">
                🚀 Coming Soon - Full Reading Tests
              </button>
            </div>
          )}

          {/* Listening Tests Tab */}
          {activeTab === 'listening' && (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">🎧</div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Listening Tests</h2>
              <p className="text-gray-600 mb-6">
                Interactive IELTS Listening tests with audio files and various question formats.
              </p>
              <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                <h3 className="font-semibold text-green-800 mb-2">Available Features:</h3>
                <ul className="text-green-700 text-left space-y-1">
                  <li>• Audio playback controls</li>
                  <li>• Form completion</li>
                  <li>• Table completion</li>
                  <li>• Note completion</li>
                  <li>• Multiple choice questions</li>
                  <li>• Matching exercises</li>
                  <li>• Real-time scoring</li>
                </ul>
              </div>
              <button className="mt-6 bg-green-600 text-white py-2 px-6 rounded-md hover:bg-green-700 transition-colors">
                🚀 Coming Soon - Full Listening Tests
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}