@echo off
cd /d "%~dp0"
python scripts\audit_files.py --root data\raw
python scripts\make_real_data_zip.py --root data\raw --output ASTRAQUANT_PS18_REAL_DATASETS.zip
if errorlevel 1 (
  echo Some dataset files are missing. See README.md for download steps.
) else (
  echo Successfully generated ASTRAQUANT_PS18_REAL_DATASETS.zip
)
pause
