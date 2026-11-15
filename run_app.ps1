Set-Location -Path $PSScriptRoot
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting AI Cervical Cancer Detection Platform..." -ForegroundColor Green
Write-Host "Automatically opening dashboard in your web browser..." -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
$py313 = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
if (Test-Path $py313) {
    & $py313 -m streamlit run app.py
} else {
    py -3.13 -m streamlit run app.py
}
