# ASTRAQUANT X 2026 — PS18 dataset acquisition bundle

## IMPORTANT: THESE ARE DOWNLOAD TOOLS, NOT THE FULL DATASETS

The four Zenodo source files (3.2 GB combined) and live Dune exports could **not** be fetched by the assistant's sandbox. No raw transaction CSVs are included in this acquisition kit. Do not mistake this ZIP for a pre-downloaded dataset bundle.

This package makes it straightforward to acquire the **genuine data** locally, preserve provenance, and create a separate ZIP of the full downloaded datasets.

## What you need (three dataset groups, four exchanges)

| Group | DEX/source | Actual dataset | Access | Labels? |
|---|---|---|---|---|
| 1 | IDEX | `IDEXTrades.csv` | Public direct Zenodo download; 2.1 GB | No independently verified fraud/organic labels |
| 1 | EtherDelta | `EtherDeltaTrades.csv` | Public direct Zenodo download; 1.1 GB | Same limitation |
| 1 | Zenodo metadata | `token_decimals.json` and `EtherDollarPrice.csv` | Public direct downloads; ~62 kB each | Not labels |
| 2 | Uniswap V3 / Ethereum | 2026 DEX swap/trade-level CSV | Dune query execution/export; account/API access | No verified fraud/organic labels |
| 3 | PancakeSwap V3 / BNB | 2026 DEX swap/trade-level CSV | Dune query execution/export; account/API access | No verified fraud/organic labels |
| Supporting | Ethereum + BNB token transfers | Transfer-event CSVs for relevant wallets | Dune query execution/export | Not a DEX trade dataset |

**Official historical record:** https://zenodo.org/records/4540223

**Zenodo direct links**:
- https://zenodo.org/records/4540223/files/IDEXTrades.csv?download=1
- https://zenodo.org/records/4540223/files/EtherDeltaTrades.csv?download=1
- https://zenodo.org/records/4540223/files/token_decimals.json?download=1
- https://zenodo.org/records/4540223/files/EtherDollarPrice.csv?download=1

**Recent DEX data catalog:** https://dune.com/data/dex-trading

**Transfer data catalog:** https://dune.com/data/transfers

**Published historical study implementation:** https://github.com/friedhelmvictor/lob-dex-wash-trading-paper

## Quick start

### Step 1 — Historical source (Python 3.10+)

From the directory containing this README, run:

```bash
python scripts/download_historical.py --output data/raw/historical
```

This downloads all 4 original Zenodo source files, resumes interrupted transfers if supported, and verifies the published MD5 checksums. These files require roughly **3.2 GB plus working space** and may take significant network time. If the source fails, open the Zenodo links above in your browser and save the files under `data/raw/historical/`.

### Step 2 — Recent Uniswap V3 and PancakeSwap V3 exports

1. Log in to Dune, and confirm your plan allows custom SQL queries and/or CSV export. Credit usage and API availability vary by plan.
2. Open new DuneSQL queries and use:
   - `sql/01_uniswap_v3_ethereum.sql`
   - `sql/02_pancakeswap_v3_bnb.sql`
   - `sql/03_token_transfers_pilot.sql`
   - `sql/04_token_transfers_bnb_pilot.sql`
3. Execute each query, inspect its returned rows and column names; adjust filters and date window when needed.
4. Export to these **exact filenames** in `data/raw/recent/`:
   - `uniswap_v3_ethereum_2026.csv`
   - `pancakeswap_v3_bnb_2026.csv`
   - `ethereum_wallet_transfers_2026.csv`
   - `bnb_wallet_transfers_2026.csv`

**Optional API export** if your account has a Dune API key: save each query, note the saved query IDs, set `DUNE_API_KEY` in your environment, and run:

```bash
python scripts/export_dune.py \
  --uniswap-query-id YOUR_QUERY_ID \
  --pancake-query-id YOUR_QUERY_ID \
  --ethereum-transfers-query-id YOUR_QUERY_ID \
  --bnb-transfers-query-id YOUR_QUERY_ID \
  --output data/raw/recent
```

The Dune API call may be unavailable or charge query credits on some accounts. The scripts only export query results; they do not generate artificial trade records. `LIMIT 100000` in the provided SQL is a pilot sample limit, **not a complete 2026 DEX history**.

### Step 3 — Verify what is present

```bash
python scripts/audit_files.py --root data/raw
```

### Step 4 — Make the ZIP containing the actual downloaded files

```bash
python scripts/make_real_data_zip.py --root data/raw --output ASTRAQUANT_PS18_REAL_DATASETS.zip
```

The packager **will refuse to produce a supposedly complete ZIP if a required file is missing**. You can use `--allow-partial` for an explicitly incomplete package; the manifest will indicate what's missing.

## How the datasets feed into the model

1. `IDEXTrades.csv` / `EtherDeltaTrades.csv`: historical transaction edges and reproducible suspicious-pattern rules from the published research. **No verified negative labels** are guaranteed.
2. Uniswap V3 2026: current AMM swap graph (wallet, swap, pool, token); temporal features and new suspicious-pattern investigations.
3. PancakeSwap V3 2026: second live DEX/protocol for cross-exchange evaluation, not automatically ground-truth-labeled by IDEX rules.
4. Supporting token transfers: observed wallet-to-wallet transfer edges for intermediate-wallet path tracing. A router or pool address is not the same as a user wallet.
5. Split chronological training/validation/test windows, log source dates and weak-label origins, build GCN + GraphSAGE and report PR-AUC, macro-F1 and false positives.
6. Apply safeguards throughout: data integrity, uncertain labels, false-positive audit, wallet pseudonymity, human-review outputs.

For more details see `docs/ASTRAQUANT_PS18_Dataset_Download_and_Usage_Guide.pdf`.

**Important limitations**: weak labels do not prove fraud; absence of a suspicious signal does not prove organic trading. These DEX transaction sources do not certify identities or common ownership of wallets. Export files are independent until normalized and validated; do not concatenate heterogeneous trade and AMM swap records blindly.
