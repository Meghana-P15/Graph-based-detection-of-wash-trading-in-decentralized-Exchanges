#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 scripts/download_historical.py --output data/raw/historical
printf '\nHistorical files verified. Export the Dune files as explained in README.md.\n'
