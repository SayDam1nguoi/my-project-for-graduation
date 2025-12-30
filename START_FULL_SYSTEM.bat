@echo off
echo ========================================
echo KHOI DONG HE THONG HOAN CHINH
echo ========================================
echo.
echo Buoc 1: Khoi dong API Server (port 8000)
echo Buoc 2: Khoi dong Web Server (port 8080)
echo.
echo ========================================
echo.

echo Dang khoi dong API Server...
start "API Server" cmd /k "python api/main.py"

timeout /t 3 /nobreak >nul

echo Dang khoi dong Web Server...
start "Web Server" cmd /k "cd frontend && python -m http.server 8080"

timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo HE THONG DA KHOI DONG!
echo ========================================
echo.
echo API Server:  http://localhost:8000
echo API Docs:    http://localhost:8000/docs
echo Web App:     http://localhost:8080/app.html
echo.
echo Nhan phim bat ky de dong cua so nay...
echo (Cac server van chay o cua so rieng)
echo ========================================
pause
