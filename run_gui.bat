@echo off
cd /d "%~dp0"
if not exist venv\Scripts\pythonw.exe (
    echo Virtual environment not found. Run setup_venv.bat first.
    pause
    exit /b 1
)
start "" venv\Scripts\pythonw.exe -m videostego gui
