@echo off
REM RAG-Lines Auto-Start Script for Windows
REM Starts API and Streamlit UI without visible command windows

setlocal enabledelayedexpansion

cls
echo ================================
echo    RAG-Lines Startup
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo [1/3] Checking LM Studio API (port 1235)...
netstat -an | find ":1235" >nul
if errorlevel 1 (
    echo WARNING: LM Studio not detected on port 1235
    echo Please start LM Studio first!
    timeout /t 2 >nul
) else (
    echo OK - LM Studio is running
)

echo [2/3] Cleaning up old processes...
REM Kill by port (more reliable than process names)
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000" ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8501" ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 2 >nul

echo [3/3] Starting services...
timeout /t 1 >nul

REM Start API in hidden window
start "RAG-API" /MIN /B python main.py api
echo API starting on port 8000...
timeout /t 4 >nul

REM Start UI in hidden window
start "RAG-UI" /MIN /B streamlit run src/ui/streamlit_app.py --logger.level=warning
echo UI starting on port 8501...
timeout /t 3 >nul

cls
echo ================================
echo ✅ RAG-Lines Started!
echo ================================
echo.
echo 📍 API:        http://localhost:8000
echo 📚 API Docs:   http://localhost:8000/docs
echo 🎨 UI:         http://localhost:8501
echo.
echo Services running in background (no visible windows).
echo.
pause
