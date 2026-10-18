@echo off
rem Double-click: run the ViperX 300 S-on-Isaac smoke check and log it to outputs\viperx300s\isaac_verify.log.
cd /d "%~dp0.."
if not exist outputs\viperx300s mkdir outputs\viperx300s
.venv\Scripts\python.exe scripts\verify_viperx300s_isaac.py > outputs\viperx300s\isaac_verify.log 2>&1
echo exit code: %ERRORLEVEL% >> outputs\viperx300s\isaac_verify.log
echo Done.
timeout /t 5
exit /b 0
