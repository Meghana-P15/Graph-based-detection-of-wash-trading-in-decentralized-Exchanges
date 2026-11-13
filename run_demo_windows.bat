@echo off
python scripts_generate_demo.py
if errorlevel 1 exit /b 1
python -m src.data --idex data/synthetic/idex.csv --etherdelta data/synthetic/etherdelta.csv --uniswap data/synthetic/uniswap.csv --pancakeswap data/synthetic/pancakeswap.csv --decimals data/synthetic/token_decimals.json --ethereum-transfers data/synthetic/ethereum_transfers.csv --bnb-transfers data/synthetic/bnb_transfers.csv
if errorlevel 1 exit /b 1
python -m src.train --source uniswap --epochs 12 --max-events 640
if errorlevel 1 exit /b 1
python -m src.infer --source uniswap --arch sage --max-events 640
if errorlevel 1 exit /b 1
streamlit run app/main.py
