@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -c "import sys, tkinter; sys.version_info >= (3, 9) or sys.exit(1); tkinter.Tcl()" >nul 2>&1
if errorlevel 1 (
    echo Python 3.9 or newer with Tcl/Tk was not found.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
"%PYTHON_EXE%" main.py
if errorlevel 1 pause
endlocal
