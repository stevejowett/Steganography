@echo off
rem Create the virtual environment and install every dependency into it.
cd /d "%~dp0"
if not exist venv (
    py -3 -m venv venv || python -m venv venv || goto :fail
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || goto :fail
echo.
echo Setup complete. Launch the GUI with run_gui.bat
exit /b 0
:fail
echo Setup failed.
exit /b 1
