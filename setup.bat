@echo off
setlocal
cd /d "%~dp0"
set "PY_CMD="
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
  set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
  set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else (
  py -3.10 --version >nul 2>nul
  if not errorlevel 1 (
    set "PY_CMD=py -3.10"
  ) else (
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 (
      set "PY_CMD=py -3.11"
    ) else (
      where python >nul 2>nul
      if not errorlevel 1 set "PY_CMD=python"
    )
  )
)
if "%PY_CMD%"=="" (
  echo Python was not found. Install Python 3.10 or 3.11 and select Add Python to PATH.
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating virtual environment with %PY_CMD%...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto failed
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Next: run run.bat
pause
exit /b 0
:failed
echo Setup did not complete. Review the error above.
pause
exit /b 1
