@echo off
title Apex-Oracle Dashboard
echo ==========================================================
echo Starting Apex-Oracle (Advanced Predictive Multi-Signal Engine)
echo ==========================================================
echo.
echo Ensure your frontend/index.html is open in another window to view predictions.
echo.

set PYTHONPATH=..
cd python_backend
python app.py
pause
