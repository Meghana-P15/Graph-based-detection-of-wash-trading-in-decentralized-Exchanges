#!/usr/bin/env bash
set -euo pipefail
python scripts_generate_demo.py
python -m src.data --idex data/synthetic/idex.csv --etherdelta data/synthetic/etherdelta.csv \
  --uniswap data/synthetic/uniswap.csv --pancakeswap data/synthetic/pancakeswap.csv \
  --decimals data/synthetic/token_decimals.json \
  --ethereum-transfers data/synthetic/ethereum_transfers.csv --bnb-transfers data/synthetic/bnb_transfers.csv
python -m src.train --source uniswap --epochs 12 --max-events 640
python -m src.infer --source uniswap --arch sage --max-events 640
printf '\nRun: streamlit run app/main.py\n'
