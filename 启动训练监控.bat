@echo off
cd /d "%~dp0"
py -3 monitor_training.py %*
if errorlevel 1 pause
