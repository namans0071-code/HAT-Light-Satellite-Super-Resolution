@echo off
title Push to GitHub - HAT-Light Satellite Super-Resolution
echo ======================================================================
echo   Pushing HAT-Light Repository to GitHub (namans0071-code)
echo ======================================================================

git remote set-url origin git@github.com-naman:namans0071-code/HAT-Light-Satellite-Super-Resolution.git
git branch -M main
echo Pushing commits and tags to GitHub...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo ======================================================================
    echo If you see "Repository not found", please ensure you have created
    echo the empty private repository on GitHub first at:
    echo https://github.com/new?name=HAT-Light-Satellite-Super-Resolution^&visibility=private
    echo ======================================================================
) else (
    echo.
    echo [SUCCESS] Repository successfully pushed to GitHub!
)
pause
