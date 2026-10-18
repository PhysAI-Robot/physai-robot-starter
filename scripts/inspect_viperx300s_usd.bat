@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe scripts\inspect_usd.py ".isaac_cache\viperx300s\vx300s\vx300s.usda" > outputs\viperx300s\usd_inspect.log 2>&1
echo exit code: %ERRORLEVEL% >> outputs\viperx300s\usd_inspect.log
exit /b 0
