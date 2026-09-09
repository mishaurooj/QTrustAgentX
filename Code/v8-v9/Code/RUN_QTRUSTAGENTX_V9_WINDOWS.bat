@echo off
setlocal

REM QTrustAgent-X v9 incremental ablations only.
REM This does NOT rerun specialists, DeBERTa, or V1-V8.

set "V8_ROOT=D:\other\QTrustAgentX\QTrustAgentX_Results_v8"
set "V9_ROOT=D:\other\QTrustAgentX\QTrustAgentX_Results_v9"

python qtrustagentx_v9_incremental_ablation.py ^
  --v8_root "%V8_ROOT%" ^
  --out_root "%V9_ROOT%" ^
  --seed 42 ^
  --repeats 5 ^
  --bootstrap_samples 2000

if errorlevel 1 (
  echo.
  echo V9 failed. Read the error above and V9_ROOT\audit\V9_PREFLIGHT_V8_CACHE.json if it exists.
  pause
  exit /b 1
)

echo.
echo V9 completed successfully.
echo Results: %V9_ROOT%
pause
endlocal
