@echo off
title Cervical Cancer AI Detection Research Platform
cd /d "%~dp0"
echo ========================================================
echo Starting AI Cervical Cancer Detection Platform...
echo Automatically opening dashboard in your web browser...
echo ========================================================
set "PY313=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if exist "%PY313%" (
    "%PY313%" -m streamlit run app.py
) else (
    py -3.13 -m streamlit run app.py
    if %ERRORLEVEL% NEQ 0 (
        python -m streamlit run app.py
    )
)
pause
