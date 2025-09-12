@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

rem ========================================
echo   Automated GA Hyperparameter Tuning
rem ========================================
echo.
echo This will help you tune GA parameters for different puzzle difficulties:
echo - Easy puzzles: Quick tuning with smaller parameters
echo - Medium puzzles: Moderate parameters and longer runs
echo - Hard puzzles: Large parameters and extended runs  
echo - Evil puzzles: Maximum parameters and very long runs
echo.
echo The script will run continuously until stopped with Ctrl+C
echo Results will be saved automatically every 10 minutes
echo.

rem Ensure we run from this script's directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

rem Prefer Anaconda Python, auto-detect if the default isn’t present
set "PYTHON_PATH=C:\Users\bagam\anaconda3\python.exe"
if not exist "%PYTHON_PATH%" (
  if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON_PATH=%USERPROFILE%\anaconda3\python.exe"
  ) else if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON_PATH=%USERPROFILE%\miniconda3\python.exe"
  ) else if exist "C:\ProgramData\anaconda3\python.exe" (
    set "PYTHON_PATH=C:\ProgramData\anaconda3\python.exe"
  ) else if exist "C:\ProgramData\miniconda3\python.exe" (
    set "PYTHON_PATH=C:\ProgramData\miniconda3\python.exe"
  ) else (
    for /f "usebackq tokens=*" %%P in (`where python 2^>NUL`) do (
      set "PYTHON_PATH=%%P"
      goto :FOUND_PY
    )
  )
)
:FOUND_PY

if not exist "%PYTHON_PATH%" (
  echo ERROR: Could not find Python. Please install Anaconda or add python to PATH.
  echo Tried: ^
   C:\Users\bagam\anaconda3\python.exe, ^
   %%USERPROFILE%%\anaconda3\python.exe, ^
   %%USERPROFILE%%\miniconda3\python.exe, ^
   C:\ProgramData\anaconda3\python.exe, ^
   C:\ProgramData\miniconda3\python.exe, and PATH.
  goto :END
)

set "PYTHONUTF8=1"
echo Using Python: "%PYTHON_PATH%"
"%PYTHON_PATH%" -c "import sys; print('Python OK:', sys.executable)" || (
  echo ERROR: Python invocation failed.
  goto :END
)

echo.
echo Starting tuning script...
echo.

"%PYTHON_PATH%" "%SCRIPT_DIR%auto_ga_tuning.py"

echo.
echo ========================================
echo Tuning completed. Check the results file for best parameters.
echo ========================================

:END
pause
endlocal
