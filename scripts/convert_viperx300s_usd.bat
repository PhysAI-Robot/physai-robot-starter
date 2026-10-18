@echo off
rem Double-click: convert the ViperX 300 S MJCF to USD with Isaac Sim and log what came out.
cd /d "%~dp0.."
if not exist outputs\viperx300s mkdir outputs\viperx300s
.venv\Scripts\python.exe scripts\convert_mjcf_to_usd.py --robot viperx300s > outputs\viperx300s\usd_convert.log 2>&1
echo exit code: %ERRORLEVEL% >> outputs\viperx300s\usd_convert.log
echo Done.
timeout /t 5
exit /b 0
