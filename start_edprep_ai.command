#!/bin/bash

# EdPrep AI - One-Click Startup Script
# Double-click this file to start your AI-powered IELTS platform

echo "🚀 Starting EdPrep AI Platform..."
echo "=================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found!"
    echo "Please make sure you're running this from the edprep-ai-prototype directory"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found!"
    echo "Please make sure you're running this from the edprep-ai-prototype directory"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Directories found"

# Start backend
echo "🔧 Starting Backend Server..."
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup first"
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate virtual environment and start backend
source venv/bin/activate
echo "✅ Virtual environment activated"

# Start backend in background
echo "🚀 Starting FastAPI backend on port 8000..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend started successfully!"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    read -p "Press Enter to exit..."
    exit 1
fi

# Go back to root directory
cd ..

# Start frontend
echo "🎨 Starting Frontend Server..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "❌ Node modules not found!"
    echo "Please run 'npm install' first"
    kill $BACKEND_PID 2>/dev/null
    read -p "Press Enter to exit..."
    exit 1
fi

# Start frontend in background
echo "🚀 Starting Next.js frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

# Check if frontend started successfully
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend started successfully!"
else
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    read -p "Press Enter to exit..."
    exit 1
fi

# Go back to root directory
cd ..

echo ""
echo "🎉 EdPrep AI Platform is now running!"
echo "=================================="
echo "🌐 Web Interface: http://localhost:3000"
echo "🔧 API Documentation: http://localhost:8000/docs"
echo "📊 Model Status: http://localhost:8000/model-status"
echo ""
echo "📝 To test your platform:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Try writing an essay and getting AI feedback"
echo "3. Check the model status at http://localhost:8000/model-status"
echo ""
echo "🛑 To stop the servers:"
echo "Press Ctrl+C or close this terminal window"
echo ""

# Keep the script running and show status
while true; do
    sleep 10
    # Check if both servers are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend server stopped unexpectedly"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend server stopped unexpectedly"
        break
    fi
done

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping EdPrep AI Platform..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    echo "Thank you for using EdPrep AI!"
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to stop
echo "Press Ctrl+C to stop the servers..."
wait


