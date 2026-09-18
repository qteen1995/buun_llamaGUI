@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=E:\Python\Python311\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0buun_launcher.pyw"
echo.
echo [exit code] %errorlevel%
pause
