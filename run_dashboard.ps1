# PowerShell launcher for Predictive Maintenance Dashboard
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Industrial Predictive Maintenance Dashboard..." -ForegroundColor Green
Write-Host "Local URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" -m streamlit run app.py
