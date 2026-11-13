#!/usr/bin/env python3
"""Package real user-downloaded files; reject incomplete packages by default."""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

FILES = [
    "historical/IDEXTrades.csv", "historical/EtherDeltaTrades.csv",
    "historical/token_decimals.json", "historical/EtherDollarPrice.csv",
    "recent/uniswap_v3_ethereum_2026.csv", "recent/pancakeswap_v3_bnb_2026.csv",
    "recent/ethereum_wallet_transfers_2026.csv", "recent/bnb_wallet_transfers_2026.csv",
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("ASTRAQUANT_PS18_REAL_DATASETS.zip"))
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    present = [n for n in FILES if (args.root / n).is_file()]
    missing = [n for n in FILES if n not in present]
    if missing and not args.allow_partial:
        print("Cannot create full dataset ZIP; missing real files:", file=sys.stderr)
        for name in missing:
            print(" -", name, file=sys.stderr)
        print("Download/export first. Use --allow-partial only for an explicitly incomplete ZIP.", file=sys.stderr)
        sys.exit(2)
    if not present:
        print("No real data files are present. Refusing to create empty dataset ZIP.", file=sys.stderr)
        sys.exit(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "complete": not missing, "included": present, "missing": missing,
                "warning": "No independent verified fraud/organic labels are supplied by these raw sources."}
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name in present:
            archive.write(args.root / name, arcname="data/raw/" + name)
            source_meta = (args.root / name).with_suffix(".source.json")
            if source_meta.is_file():
                archive.write(source_meta, arcname="data/raw/" + source_meta.relative_to(args.root).as_posix())
            print("Included", name)
        archive.writestr("DATASET_MANIFEST.json", json.dumps(manifest, indent=2))
    print("Created:", args.output, "complete=" + str(not missing))

if __name__ == "__main__":
    main()
