@echo off
cd /d "D:\RunPodMediaConsoleWeb"
call ".venv\Scripts\activate.bat"
python -m uvicorn app:app --host 127.0.0.1 --port 8765
pause
