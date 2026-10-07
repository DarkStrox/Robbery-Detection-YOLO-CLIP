@echo off
echo =========================================================================
echo  Starting Robbery Detection System (YOLO + Crop Super-Res + CLIP + VLM)
echo =========================================================================
cd /d "%~dp0"
streamlit run app_ui.py
pause
