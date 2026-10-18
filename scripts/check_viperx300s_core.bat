@echo off
rem Double-click: format, lint, import-linter and the ViperX 300 S/registration tests. Log: outputs\viperx300s\core_check.log
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
set "LOG=outputs\viperx300s\core_check.log"
echo ==== started %date% %time% > "%LOG%"
"%PY%" -m ruff format src\physai\sim\isaac\description.py src\physai\robots\viperx300s src\physai\robots\so101\isaac_env.py tests\core\acceptance\viperx300s research\tests\imitation_learning\test_plugin_registration.py scripts\verify_viperx300s_isaac.py scripts\convert_mjcf_to_usd.py scripts\inspect_usd.py scripts\diag_viperx300s_render.py >> "%LOG%" 2>&1
echo ---- ruff check >> "%LOG%"
"%PY%" -m ruff check src\physai\sim\isaac\description.py src\physai\robots\viperx300s tests\core\acceptance\viperx300s scripts\verify_viperx300s_isaac.py scripts\convert_mjcf_to_usd.py scripts\inspect_usd.py scripts\diag_viperx300s_render.py >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"
echo ---- import-linter >> "%LOG%"
.venv\Scripts\lint-imports.exe >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"
echo ---- tests >> "%LOG%"
"%PY%" -m pytest tests\core\acceptance\viperx300s tests\core\boundaries tests\core\unit -q -p no:cacheprovider --tb=short >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"
"%PY%" -m pytest research\tests\imitation_learning\test_plugin_registration.py -q -p no:cacheprovider --tb=short >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"
echo ==== finished >> "%LOG%"
exit /b 0
