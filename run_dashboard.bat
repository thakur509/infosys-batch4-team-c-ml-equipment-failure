@echo off
TITLE Predictive Maintenance Dashboard
echo ============================================================
echo Starting Industrial Predictive Maintenance Dashboard...
echo Local URL: http://localhost:8501
echo ============================================================
.\.venv\Scripts\python.exe -m streamlit run app.py
pause
