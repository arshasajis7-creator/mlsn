@echo off
REM run_merge.bat
REM This batch file launches PowerShell with an execution policy bypass to run merge_contents.ps1
REM Place this .bat in the same folder as merge_contents.ps1 and double-click it.

setlocal

REM Resolve script folder (so the .bat works even when started from other cwd)
set "SCRIPT_DIR=%~dp0"
REM Trim trailing backslash if any
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Build full path to PS1
set "PS1=%SCRIPT_DIR%\merge_contents.ps1"

REM Check the PS1 exists
if not exist "%PS1%" (
  echo ERROR: Could not find "%PS1%". Make sure merge_contents.ps1 is in the same folder as this .bat.
  pause
  endlocal
  exit /b 1
)

REM Run PowerShell (no profile, bypass execution policy) and wait for it to finish; window will close when done.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%PS1%'"
endlocal
