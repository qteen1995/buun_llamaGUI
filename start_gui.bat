@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "HERE=%~dp0"

rem 首选：已知带 tkinter 的 Python
if exist "E:\Python\Python311\pythonw.exe" (
  start "" "E:\Python\Python311\pythonw.exe" "%HERE%buun_launcher.pyw"
  exit /b 0
)

rem 其次：PATH 里的 pythonw
where pythonw >nul 2>nul
if not errorlevel 1 (
  start "" pythonw "%HERE%buun_launcher.pyw"
  exit /b 0
)

echo [ERROR] pythonw.exe not found.
echo Please install Python 3.10+ (with tkinter) and run:
echo     pythonw "%HERE%buun_launcher.pyw"
pause
exit /b 1
