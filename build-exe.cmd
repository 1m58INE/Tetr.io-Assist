@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 goto fail
)
REM Fail before PyInstaller if a local Python module was omitted from the ZIP.
.venv\Scripts\python.exe verify_package.py
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto fail
REM Build-time tool only: runtime application has no additional dependencies.
.venv\Scripts\python.exe -m pip install --only-binary=:all: "PyInstaller>=6,<7"
if errorlevel 1 goto fail
REM Run the real regression suite in the same interpreter used for PyInstaller.
.venv\Scripts\python.exe -m unittest discover -q
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name TetrioAssist --hidden-import observed_attack_sequence --hidden-import attack_structure11113 --hidden-import attack_rate --hidden-import telemetry --hidden-import combo_rules --hidden-import combat_rules --hidden-import opening --hidden-import performance_budget --hidden-import legacy_quarantine --hidden-import pressure_refresh --hidden-import legacy_integrity --hidden-import garbage_tracking --hidden-import config_security --hidden-import input_rhythm --hidden-import color_vision --hidden-import piece_vision --hidden-import vision_failure --hidden-import spawn_vision --hidden-import grid_calibration --hidden-import window_roi --hidden-import garbage_transition --hidden-import tspin_factory --hidden-import b2b_search --hidden-import competitive_policy --hidden-import offensive_beam --hidden-import adaptive_opener --hidden-import canonical_openers --hidden-import opener_commit --hidden-import perfect_clear_planner --hidden-import vision_recovery --hidden-import offense_runtime --hidden-import chain_director --hidden-import stall_monitor --hidden-import tspin_initiator --hidden-import transition_guard --hidden-import tbp_bridge --hidden-import engine_policy --hidden-import bundled_tbp_bot assist.py
if errorlevel 1 goto fail
REM The main --windowed EXE has no reliable stdin/stdout. A separate console
REM subprocess is required for the self-contained TBP adviser. No Rust needed.
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --console --name TetrioBundledBot --hidden-import observed_attack_sequence --hidden-import attack_structure11113 --hidden-import attack_rate --hidden-import performance_budget --hidden-import engine --hidden-import motion --hidden-import offensive_beam --hidden-import combo_rules --hidden-import combat_rules --hidden-import competitive_policy --hidden-import tspin_factory --hidden-import b2b_search --hidden-import chain_director --hidden-import tspin_initiator --hidden-import tbp_bridge bundled_tbp_bot.py
if errorlevel 1 goto fail
.venv\Scripts\python.exe bundled_check.py --exe dist\TetrioBundledBot.exe
if errorlevel 1 goto fail
REM Randomized marker + exit-code prevents accepting another build's stale marker.
for /f "delims=" %%A in ('.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(16))"') do set "SMOKE_TOKEN=%%A"
if not defined SMOKE_TOKEN goto fail
set "SMOKE_MARKER=%TEMP%\TetrioAssist_smoke_%SMOKE_TOKEN%.txt"
if exist "%SMOKE_MARKER%" del "%SMOKE_MARKER%"
start /wait "" "dist\TetrioAssist.exe" --self-test=%SMOKE_TOKEN%
if errorlevel 1 goto fail
if not exist "%SMOKE_MARKER%" (
  echo ERROR: EXE failed packaged startup smoke test.
  goto fail
)
findstr /X /C:"%SMOKE_TOKEN%" "%SMOKE_MARKER%" >nul
if errorlevel 1 goto fail
del "%SMOKE_MARKER%"
REM Frozen GUI resolves CC2 from dist\external_solver, not the source folder.
REM Re-check the actual TBP route before copying, but never fail an EXE build
REM merely because optional Cold Clear 2 is absent/incompatible.
.venv\Scripts\python.exe stage_coldclear2.py --optional
if errorlevel 1 goto fail
echo Created and smoke-checked: dist\TetrioAssist.exe and dist\TetrioBundledBot.exe
pause
exit /b 0
:fail
echo Build failed. Read the error above.
pause
exit /b 1
