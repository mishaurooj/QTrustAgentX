@echo off
setlocal

REM ================================================================
REM QTrustAgent-X corrective experiments
REM Edit the four paths below before running.
REM ================================================================

set PYTHON=python
set DATA_ROOT=D:\other\QTrustAgentX\Dataset_Reorganized
set V8_ROOT=D:\other\QTrustAgentX\QTrustAgentX_Results_v8
set OUT_ROOT=D:\other\QTrustAgentX\QTrustAgentX_Corrective
set INCIDENT_EVIDENCE=D:\other\QTrustAgentX\QTrustAgentX_Results_v8\predictions\INCIDENT_evidence_table.csv
set SEED=42

%PYTHON% qtrustagentx_corrective_experiments.py ^
  --data_root "%DATA_ROOT%" ^
  --v8_root "%V8_ROOT%" ^
  --out_root "%OUT_ROOT%" ^
  --incident_evidence "%INCIDENT_EVIDENCE%" ^
  --seed %SEED% ^
  --run all ^
  --llm_model microsoft/deberta-v3-base ^
  --llm_ensemble 3 ^
  --llm_epochs 6 ^
  --llm_patience 2 ^
  --llm_batch_size 8 ^
  --llm_grad_accum 4 ^
  --llm_max_length 384 ^
  --llm_lr 1e-5 ^
  --phase_coeffs 0.5 1.0 ^
  --pair_strength 0.35

if errorlevel 1 (
  echo.
  echo Corrective experiment failed. Check the terminal output and OUT_ROOT\audit.
  exit /b 1
)

echo.
echo Completed. Paper tables are under:
echo   %OUT_ROOT%\tables
pause
