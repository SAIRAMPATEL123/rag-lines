@echo off
title RAG-Lines Launcher
color 0A

set PYTHON="C:\Users\sairam.bathula\Desktop\install p\python.exe"
set STREAMLIT="C:\Users\sairam.bathula\Desktop\install p\Scripts\streamlit.exe"
set PROJECT="C:\Users\sairam.bathula\Desktop\r coidel\rag-development\rag-lines\rag-lines"

echo ============================================
echo        RAG-Lines - Starting App...
echo ============================================
echo.

:: Check if Ollama is installed
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not installed or not in PATH.
    echo Please download and install from: https://ollama.ai
    echo.
    pause
    exit /b 1
)

:: Start Ollama serve in background
echo [1/3] Starting Ollama server...
start "Ollama Server" cmd /k "ollama serve"
timeout /t 3 /nobreak >nul

:: Start FastAPI backend
echo [2/3] Starting FastAPI backend on http://localhost:8000 ...
start "RAG-Lines API" cmd /k "cd /d %PROJECT% && %PYTHON% main.py api"
timeout /t 4 /nobreak >nul

:: Start Streamlit UI
echo [3/3] Starting Streamlit UI on http://localhost:8501 ...
start "RAG-Lines UI" cmd /k "cd /d %PROJECT% && %STREAMLIT% run src\ui\streamlit_app.py"
timeout /t 3 /nobreak >nul

echo.
echo ============================================
echo  All services started!
echo  UI  : http://localhost:8501
echo  API : http://localhost:8000/docs
echo ============================================
echo.
echo  Press any key to open UI in browser...
pause >nul
start http://localhost:8501
