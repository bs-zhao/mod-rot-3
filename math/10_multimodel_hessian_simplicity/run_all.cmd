@echo off
setlocal
REM Run from any directory; always resolve repository root.
cd /d "%~dp0\..\.."
python math\10_multimodel_hessian_simplicity\run.py --models C1 C2 C3 CR3 %*
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" echo Hessian atlas failed with exit code %ERR%.
exit /b %ERR%
