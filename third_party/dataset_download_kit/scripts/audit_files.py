#!/usr/bin/env python3
"""Inspect expected real datasets; no labels are inferred."""
import argparse
import csv
import json
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
    args = p.parse_args()
    report = []
    for name in FILES:
        file = args.root / name
        item = {"file": name, "present": file.is_file(),
                "bytes": file.stat().st_size if file.is_file() else 0}
        if file.is_file() and file.suffix == ".csv":
            with open(file, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                item["header"] = next(csv.reader(f), [])
        report.append(item)
        print(("OK " if item["present"] else "MISSING ") + name +
              (f" ({item['bytes']:,} B)" if item["present"] else ""))
    out = args.root / "audit.json"
    args.root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("Audit saved to", out)

if __name__ == "__main__":
    main()
