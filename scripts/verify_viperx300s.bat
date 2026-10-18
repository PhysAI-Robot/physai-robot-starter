@echo off
rem Double-click to verify the ViperX 300 S robot end to end. Everything is logged to
rem outputs\viperx300s\verify.log (and diag.log); nothing here needs typing. Safe to re-run.
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
set "LOG=outputs\viperx300s\verify.log"
if not exist outputs\viperx300s mkdir outputs\viperx300s
echo ==== ViperX 300 S verification started %date% %time% > "%LOG%"
echo Verifying the ViperX 300 S robot. This takes a few minutes; the window closes itself when done.

echo ---- format the files this work added or touched
echo ---- format the files this work added or touched >> "%LOG%"
"%PY%" -m ruff format src\physai\robots\viperx300s src\physai\robots\registry.py src\physai\policy research\scripted_experts research\scripts\viperx300s_check.py research\scripts\viperx300s_diag.py research\tests\imitation_learning\test_plugin_registration.py tests\core\acceptance\viperx300s tests\conftest.py scripts\fetch_assets.py >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"

echo ---- lint those files
echo ---- lint those files >> "%LOG%"
"%PY%" -m ruff check src\physai\robots\viperx300s research\scripted_experts\viperx300s_pick_place_expert.py research\scripts\viperx300s_check.py research\scripts\viperx300s_diag.py tests\core\acceptance\viperx300s >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"

echo ---- viperx300s tests
echo ---- viperx300s tests >> "%LOG%"
"%PY%" -m pytest tests\core\acceptance\viperx300s -q -p no:cacheprovider --tb=short >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"

echo ---- research tests
echo ---- research tests >> "%LOG%"
"%PY%" -m pytest research\tests -q -p no:cacheprovider --tb=short -m "not isaac" >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"

echo ---- so101 expert still 100% (20 seeds, default manifest)
echo ---- so101 expert still 100% (20 seeds, default manifest) >> "%LOG%"
"%PY%" scripts\eval_policy.py --manifest research\configs\so101_single_cube_fixed_place.yaml --policy scripted --episodes 20 --seed 0 --max-steps 500 >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"

echo ---- viperx300s expert gate after formatting (100 seeds, 1000-1099)
echo ---- viperx300s expert gate after formatting (100 seeds, 1000-1099) >> "%LOG%"
"%PY%" scripts\eval_policy.py --manifest research\configs\viperx300s_pick_place.yaml --policy scripted --episodes 100 --seed 1000 --max-steps 600 >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"

echo ==== ViperX 300 S verification finished %date% %time% >> "%LOG%"
echo Done.
timeout /t 5
exit /b 0
