@echo off
echo ========================================
echo RESTART API - XOA CACHE
echo ========================================
echo.

echo Buoc 1: Tat Python processes...
taskkill /IM python.exe /F 2>nul
timeout /t 2 /nobreak >nul

echo Buoc 2: Xoa cache Python...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Cache da duoc xoa!

echo.
echo Buoc 3: Khoi dong API server...
echo.
echo ========================================
echo API Server dang chay...
echo ========================================
echo.

python api/main.py

pause
