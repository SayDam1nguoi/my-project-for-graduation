@echo off
echo ========================================
echo Starting Web Server
echo ========================================
echo.
echo Server se chay tai: http://localhost:8080
echo.
echo Nhan Ctrl+C de dung server
echo.
echo ========================================
echo.

cd frontend
python -m http.server 8080

pause
