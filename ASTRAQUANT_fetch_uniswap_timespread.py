"""Free, wallet-free Uniswap v3 USDC/WETH (0.05%) Ethereum swap pilot.

Usage from ASTRAQUANT project root:
    python -m pip install requests
    python ASTRAQUANT_fetch_uniswap_timespread.py --limit 300 --blocks 1000

This collects *real* on-chain swap events, not fraud labels. JSON-RPC provider
is a shared public service and can impose rate limits or restrict eth_getLogs.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

RPC = "https://ethereum-rpc.publicnode.com"
POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # Ethereum Uniswap v3 USDC/WETH 0.05%
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"  # token0, 6 decimals
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"  # token1, 18 decimals
COLUMNS = ["blockchain", "project", "version", "block_time", "block_number", "tx_hash",
           "evt_index", "tx_from", "tx_to", "taker", "maker", "project_contract_address",
           "token_bought_address", "token_sold_address", "token_bought_symbol", "token_sold_symbol",
           "token_bought_amount", "token_sold_amount", "amount_usd"]


def rpc(session, url, method, params):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    last_error = None
    for attempt in range(4):
        try:
            response = session.post(url, json=payload, timeout=25)
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(str(body["error"]))
            if "result" not in body:
                raise RuntimeError("JSON-RPC response missing result")
            return body["result"]
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{method} failed: {last_error}")


def signed256(hex_word):
    n = int(hex_word, 16)
    return n - (1 << 256) if n >= (1 << 255) else n


def normalized_amounts(log):
    data = log["data"][2:]
    if len(data) < 128:
        raise ValueError("Invalid Swap event amount payload")
    a0 = signed256(data[:64])
    a1 = signed256(data[64:128])
    # amount > 0 means token goes TO pool; amount < 0 means token leaves pool
    if a0 > 0 and a1 < 0:
        return USDC, WETH, "USDC", "WETH", a0 / 1e6, -a1 / 1e18
    if a1 > 0 and a0 < 0:
        return WETH, USDC, "WETH", "USDC", a1 / 1e18, -a0 / 1e6
    return None  # unexpected/zero amounts


def collect(limit, blocks, rpc_url):
    with requests.Session() as s:
        latest = int(rpc(s, rpc_url, "eth_blockNumber", []), 16)
        end = latest - 12  # exclude newest blocks to reduce reorg risk
        start = max(0, end - blocks + 1)
        print(f"Reading real swaps from Ethereum blocks {start} to {end}")
        seen = set()
        logs = []
        # Scan the ENTIRE requested range, oldest to newest. Selecting first N
        # swaps from the newest blocks would destroy the intended time coverage.
        for lo in range(start, end + 1, 10):
            hi = min(end, lo + 9)
            part = rpc(s, rpc_url, "eth_getLogs", [{"fromBlock": hex(lo), "toBlock": hex(hi),
                     "address": POOL, "topics": [SWAP_TOPIC]}])
            for log in part:
                key = (log["transactionHash"].lower(), int(log["logIndex"], 16))
                if key not in seen:
                    seen.add(key)
                    logs.append(log)
            time.sleep(0.15)
        if not logs:
            raise RuntimeError("No swap logs in scanned blocks. Increase --blocks or try again later.")
        logs.sort(key=lambda x: (int(x["blockNumber"], 16), int(x["logIndex"], 16)))
        total_found = len(logs)
        # Uniform rank-based sampling across the whole block interval.
        # This is a time-spread exploratory sample, not a complete trading history.
        if len(logs) > limit:
            if limit == 1:
                logs = [logs[len(logs) // 2]]
            else:
                indices = [(i * (len(logs) - 1)) // (limit - 1) for i in range(limit)]
                logs = [logs[i] for i in indices]
        print(f"Found {total_found} swaps across {blocks} blocks; selected {len(logs)} time-spread events")
        block_cache = {}
        tx_cache = {}
        rows = []
        for idx, log in enumerate(logs, start=1):
            amounts = normalized_amounts(log)
            if amounts is None:
                continue
            buy_or_sell = amounts
            bn = log["blockNumber"]
            tx_hash = log["transactionHash"]
            if bn not in block_cache:
                block_cache[bn] = rpc(s, rpc_url, "eth_getBlockByNumber", [bn, False])
            if tx_hash not in tx_cache:
                tx_cache[tx_hash] = rpc(s, rpc_url, "eth_getTransactionByHash", [tx_hash])
            tx = tx_cache[tx_hash]
            if not tx or not tx.get("from"):
                print(f"Skipping tx without origin: {tx_hash}")
                continue
            timestamp = datetime.fromtimestamp(int(block_cache[bn]["timestamp"], 16), tz=timezone.utc).isoformat()
            sold_token, bought_token, sold_sym, bought_sym, amt_in, amt_out = buy_or_sell
            rows.append({"blockchain": "ethereum", "project": "uniswap", "version": 3,
                "block_time": timestamp, "block_number": int(bn, 16), "tx_hash": tx_hash,
                "evt_index": int(log["logIndex"], 16), "tx_from": tx["from"], "tx_to": tx.get("to") or "",
                "taker": "", "maker": "", "project_contract_address": POOL,
                "token_bought_address": bought_token, "token_sold_address": sold_token,
                "token_bought_symbol": bought_sym, "token_sold_symbol": sold_sym,
                "token_bought_amount": f"{amt_out:.12g}", "token_sold_amount": f"{amt_in:.12g}",
                "amount_usd": ""})  # USD price NOT verified; do not fabricate it
            if idx % 20 == 0:
                print(f"Decoded {idx}/{len(logs)} swaps")
            time.sleep(0.08)
        return rows, (start, end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="maximum swap events to fetch (1-500)")
    ap.add_argument("--blocks", type=int, default=200, help="recent Ethereum blocks to scan (10-1000)")
    ap.add_argument("--rpc", default=RPC, help="Ethereum mainnet JSON-RPC URL")
    ap.add_argument("--output", default="data/raw/recent/uniswap_v3_time_spread.csv")
    args = ap.parse_args()
    if not 1 <= args.limit <= 500 or not 10 <= args.blocks <= 1000:
        ap.error("--limit must be 1-500 and --blocks must be 10-1000")
    rows, range_used = collect(args.limit, args.blocks, args.rpc)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["block_time"], r["tx_hash"], r["evt_index"])))
    print(f"SUCCESS: {len(rows)} real on-chain Uniswap V3 swaps saved to {path}")
    print(f"Block window: {range_used}; source: {args.rpc}")
    if rows:
        stamps = [r["block_time"] for r in rows]
        print(f"Collected timestamp range: {min(stamps)} through {max(stamps)}")
    print("Sampled events are NOT a complete continuous trading history: temporal event-count features may be biased.")
    print("USD values intentionally blank; transaction origin != proven user identity.")
    print("Do not present these records as confirmed fraud or legitimate trading.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, requests.RequestException, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
