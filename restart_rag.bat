@echo off
REM RAG-Lines Restart Script - Kills and restarts everything

cls
echo ================================
echo    RAG-Lines Restart
echo ================================
echo.

echo Stopping all services...
REM Kill by port (more reliable)
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000" ^| find "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
    echo API stopped (port 8000)
)
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8501" ^| find "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
    echo UI stopped (port 8501)
)

timeout /t 2 >nul
echo.
echo Starting fresh...
timeout /t 1 >nul

call run_rag.bat
