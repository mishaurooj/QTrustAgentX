@echo off
setlocal
cd /d "%~dp0"
if not exist "E:\other\QTrustAgentX\Dataset_Reorganized.zip" (
  echo Missing E:\other\QTrustAgentX\Dataset_Reorganized.zip
  echo Place the ORIGINAL uploaded archive there and run again.
  exit /b 1
)
python 01_build_dataset.py --zip "E:\other\QTrustAgentX\Dataset_Reorganized.zip" --out "E:\other\QTrustAgentX\QTrustAgentX_Validation\data"
if errorlevel 1 exit /b 1
python 02_train_evaluate.py --data "E:\other\QTrustAgentX\QTrustAgentX_Validation\data"
if errorlevel 1 exit /b 1
python 03_export_and_verify_circuits.py --experiment "E:\other\QTrustAgentX\QTrustAgentX_Validation\data\experiment"
if errorlevel 1 exit /b 1
python 04_make_figures_and_report.py --data "E:\other\QTrustAgentX\QTrustAgentX_Validation\data"
if errorlevel 1 exit /b 1
python 05_ablation_study.py --data "E:\other\QTrustAgentX\QTrustAgentX_Validation\data"
if errorlevel 1 exit /b 1
python 06_make_latex_ablation_tables.py
if errorlevel 1 exit /b 1
python 07_eval_original_four_qubits.py --data "E:\other\QTrustAgentX\QTrustAgentX_Validation\data"
if errorlevel 1 exit /b 1
python 08_make_quantum_control_latex.py
if errorlevel 1 exit /b 1
echo Completed. See data\experiment\VALIDATION_REPORT.md and ABLATION_STUDY_FULL_TABLES.md
endlocal
