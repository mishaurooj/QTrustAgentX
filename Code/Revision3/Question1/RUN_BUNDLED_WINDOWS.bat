@echo off
setlocal
cd /d "%~dp0"
echo Re-running with the generated DATA already present in this package.
if not exist "data\multimodal_31_combinations_1000_each.csv" (
  echo Missing bundled dataset. Extract the entire ZIP first.
  exit /b 1
)
python 02_train_evaluate.py --data "%~dp0data"
if errorlevel 1 exit /b 1
python 03_export_and_verify_circuits.py --experiment "%~dp0data\experiment"
if errorlevel 1 exit /b 1
python 04_make_figures_and_report.py --data "%~dp0data"
if errorlevel 1 exit /b 1
python 05_ablation_study.py --data "%~dp0data"
if errorlevel 1 exit /b 1
python 06_make_latex_ablation_tables.py
if errorlevel 1 exit /b 1
python 07_eval_original_four_qubits.py --data "%~dp0data"
if errorlevel 1 exit /b 1
python 08_make_quantum_control_latex.py
if errorlevel 1 exit /b 1
echo Complete. See data\experiment\ABLATION_STUDY_FULL_TABLES.md
endlocal
