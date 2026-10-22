Set-Location -Path $PSScriptRoot
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting AI Cervical Cancer Detection Platform..." -ForegroundColor Green
Write-Host "Automatically opening dashboard in your web browser..." -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
py -3.13 -m streamlit run app.py
