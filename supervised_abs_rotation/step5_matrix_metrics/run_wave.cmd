@echo off
setlocal
call conda activate py11
if errorlevel 1 exit /b 1
cd /d "%~dp0\..\.."
python supervised_abs_rotation/step5_matrix_metrics/run_batch.py %*
exit /b %errorlevel%
