#!/usr/bin/env python3
"""Run user's saved Dune queries and export real query results as CSV."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = "https://api.dune.com/api/v1"


def request_json(method, path, key, data=None):
    headers = {"X-Dune-API-Key": key, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(ROOT + path, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=75) as r:
        return json.load(r)


def export_query(query_id, target, key, timeout):
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing {target}; move it first")
    print(f"[EXECUTE] query #{query_id}")
    response = request_json("POST", f"/query/{query_id}/execute", key, {})
    execution_id = response["execution_id"]
    started = time.monotonic()
    while True:
        status = request_json("GET", f"/execution/{execution_id}/status", key)
        state = status.get("state", "UNKNOWN")
        print(f"[STATUS] {execution_id}: {state}")
        if state in ("QUERY_STATE_COMPLETED", "QUERY_STATE_SUCCEEDED"):
            break
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"):
            raise RuntimeError(f"Query not completed: {status}")
        if time.monotonic() - started >= timeout:
            raise TimeoutError(f"Query timeout {execution_id}")
        time.sleep(8)

    req = urllib.request.Request(
        ROOT + f"/execution/{execution_id}/results/csv",
        headers={"X-Dune-API-Key": key, "Accept": "text/csv"},
    )
    part = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as response, open(part, "wb") as output:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    if part.stat().st_size == 0:
        raise ValueError(f"Downloaded an empty result from query {query_id}")
    part.rename(target)
    metadata = {"query_id": query_id, "execution_id": execution_id,
                "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "filename": target.name, "bytes": target.stat().st_size}
    target.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2))
    print(f"[SAVED] {target} ({target.stat().st_size:,} bytes)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uniswap-query-id", type=int)
    p.add_argument("--pancake-query-id", type=int)
    p.add_argument("--ethereum-transfers-query-id", type=int)
    p.add_argument("--bnb-transfers-query-id", type=int)
    p.add_argument("--output", type=Path, default=Path("data/raw/recent"))
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    api_key = os.environ.get("DUNE_API_KEY", "")
    if not api_key:
        p.error("Set DUNE_API_KEY in your environment; do not hard-code it in a file")
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = [
        (args.uniswap_query_id, "uniswap_v3_ethereum_2026.csv"),
        (args.pancake_query_id, "pancakeswap_v3_bnb_2026.csv"),
        (args.ethereum_transfers_query_id, "ethereum_wallet_transfers_2026.csv"),
        (args.bnb_transfers_query_id, "bnb_wallet_transfers_2026.csv"),
    ]
    if not any(qid for qid, _ in mapping):
        p.error("Supply at least one --*-query-id argument")
    failures = []
    for qid, file in mapping:
        if qid:
            try:
                export_query(qid, args.output / file, api_key, args.timeout)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError,
                    RuntimeError, KeyError, TimeoutError, ValueError) as exc:
                print(f"[ERROR] {file}: {exc}", file=sys.stderr)
                failures.append(file)
    if failures:
        print("[FAILED] " + ", ".join(failures), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
