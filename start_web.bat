@echo off
REM Start Web Application - Interview Analysis System
REM Simple batch file to start API and open frontend

echo.
echo ========================================================
echo   INTERVIEW ANALYSIS SYSTEM - WEB VERSION
echo ========================================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found
echo.

REM Start API in new window
echo Starting API Backend...
echo Location: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo.
start "Interview Analysis API" cmd /k "cd /d %~dp0api && python main.py"

REM Wait for API to start
echo Waiting for API to start (10 seconds)...
timeout /t 10 /nobreak >nul

REM Open Frontend
echo Opening Frontend...
start "" "%~dp0frontend\app.html"

echo.
echo ========================================================
echo   WEB APPLICATION IS RUNNING!
echo ========================================================
echo.
echo Status:
echo   - API Backend: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo   - Frontend: Opened in browser
echo.
echo Quick Start:
echo   1. Click tab: Camera Truc Tiep
echo   2. Click button: Bat Camera
echo   3. Allow camera permissions
echo   4. Wait for AI models (~5-10 sec)
echo   5. See real-time emotion detection!
echo.
echo To stop:
echo   - Close the API window
echo   - Or press CTRL+C in API window
echo.
echo Documentation: HUONG_DAN_CHAY_WEB.md
echo.
echo ========================================================
echo   Enjoy analyzing!
echo ========================================================
echo.

pause
