@echo off
echo ========================================================
echo Starting Stock Price Prediction Streamlit Dashboard...
echo ========================================================
echo.
echo Checking dependencies...
python -c "import streamlit, yfinance, pandas, numpy, sklearn, matplotlib, seaborn, plotly" >nul 2>&1
if %errorlevel% neq 0 (
    echo Dependencies not fully installed! Running installation...
    pip install -r requirements.txt
) else (
    echo All dependencies satisfied!
)
echo.
echo Launching Streamlit dashboard...
streamlit run app.py
pause
