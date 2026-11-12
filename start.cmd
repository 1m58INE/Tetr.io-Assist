@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe verify_package.py
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe assist.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo Failed. Check the error above.
pause
exit /b 1
