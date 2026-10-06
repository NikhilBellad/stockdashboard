@echo off
cd /d "%~dp0"
REM Uses the signed python.exe only (no pip.exe / streamlit.exe launchers, which Smart App Control blocks)
python -m pip install --user -r requirements.txt
python -m streamlit run app.py
pause
