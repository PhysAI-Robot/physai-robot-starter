@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe scripts\eval_policy.py --sim isaac --manifest configs\manifests\so101_single_cube_fixed_place.yaml --policy constant --episodes 1 --max-steps 20 > outputs\viperx300s\control2_head.log 2>&1
echo exit code: %ERRORLEVEL% >> outputs\viperx300s\control2_head.log
exit /b 0
