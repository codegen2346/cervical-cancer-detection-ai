@echo off
title Cervical Cancer AI Detection Research Platform
cd /d "%~dp0"
echo ========================================================
echo Starting AI Cervical Cancer Detection Platform...
echo Automatically opening dashboard in your web browser...
echo ========================================================
py -3.13 -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Retrying with default python command...
    python -m streamlit run app.py
)
pause
