@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
echo.
echo RunPod Media Console Web will listen on all local network interfaces.
echo Use http://localhost:8765 on this PC, or http://YOUR-LAN-IP:8765 from another device.
echo Remote devices must sign in with the credentials below.
echo Username: runpod
python -c "import app; print('Password: ' + app.LAN_ACCESS_TOKEN)"
echo If Windows Firewall asks, allow access on Private networks.
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 8765
pause
