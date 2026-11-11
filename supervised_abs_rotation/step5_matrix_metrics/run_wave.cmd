@echo off
setlocal
call conda activate py11
if errorlevel 1 exit /b 1
cd /d "%~dp0\..\.."
for %%M in (C1 C2 C3 CR3) do (
    python supervised_abs_rotation/step5_matrix_metrics/run.py --model %%M
    if errorlevel 1 exit /b 1
)
python supervised_abs_rotation/step5_matrix_metrics/run.py --compare
exit /b %errorlevel%
