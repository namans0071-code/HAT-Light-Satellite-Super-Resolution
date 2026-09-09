@echo off
title HAT-Light Satellite Super-Resolution Studio
echo ======================================================================
echo   Launching HAT-Light Satellite Super-Resolution Studio
echo ======================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" app.py
) else (
    python app.py
)

pause
