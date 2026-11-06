@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment not found. Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m uvicorn app:app --host 127.0.0.1 --port 8000
