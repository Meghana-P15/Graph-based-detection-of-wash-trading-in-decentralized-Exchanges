@echo off
cd /d "%~dp0"
python scripts\download_historical.py --output data\raw\historical
if errorlevel 1 (
  echo Download incomplete. Check your internet connection and try again.
  pause
  exit /b 1
)
echo Historical downloads verified. Export the Dune datasets using README.md.
pause
