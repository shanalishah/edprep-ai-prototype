@echo off
REM EdPrep AI - One-Click Startup Script for Windows
REM Double-click this file to start your AI-powered IELTS platform

echo 🚀 Starting EdPrep AI Platform...
echo ==================================

REM Get the directory where this script is located
cd /d "%~dp0"

echo 📁 Working directory: %CD%

REM Check if backend directory exists
if not exist "backend" (
    echo ❌ Backend directory not found!
    echo Please make sure you're running this from the edprep-ai-prototype directory
    pause
    exit /b 1
)

REM Check if frontend directory exists
if not exist "frontend" (
    echo ❌ Frontend directory not found!
    echo Please make sure you're running this from the edprep-ai-prototype directory
    pause
    exit /b 1
)

echo ✅ Directories found

REM Start backend
echo 🔧 Starting Backend Server...
cd backend

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run the setup first
    pause
    exit /b 1
)

REM Activate virtual environment and start backend
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Start backend in background
echo 🚀 Starting FastAPI backend on port 8000...
start /b uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Go back to root directory
cd ..

REM Start frontend
echo 🎨 Starting Frontend Server...
cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo ❌ Node modules not found!
    echo Please run 'npm install' first
    pause
    exit /b 1
)

REM Start frontend in background
echo 🚀 Starting Next.js frontend on port 3000...
start /b npm run dev

REM Wait a moment for frontend to start
timeout /t 5 /nobreak > nul

REM Go back to root directory
cd ..

echo.
echo 🎉 EdPrep AI Platform is now running!
echo ==================================
echo 🌐 Web Interface: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 Model Status: http://localhost:8000/model-status
echo.
echo 📝 To test your platform:
echo 1. Open http://localhost:3000 in your browser
echo 2. Try writing an essay and getting AI feedback
echo 3. Check the model status at http://localhost:8000/model-status
echo.
echo 🛑 To stop the servers:
echo Close this terminal window or press Ctrl+C
echo.

pause

