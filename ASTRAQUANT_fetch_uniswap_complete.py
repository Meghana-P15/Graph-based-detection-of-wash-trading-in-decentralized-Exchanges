"""ASTRAQUANT: complete bounded-window, three-pool Uniswap v3 Swap collector.

Run from the project root:
    python -m pip install requests
    python ASTRAQUANT_fetch_uniswap_complete.py --blocks 120
    python ASTRAQUANT_fetch_uniswap_complete.py --blocks 120 --end-block 26406792

Output is suitable for `python -m src.data --uniswap <csv> ...`.
No wallet/API key required. Requires access to a public Ethereum JSON-RPC provider.
The file contains on-chain events, NOT confirmed fraud labels. USD left blank.

Completeness claim is restricted to the specified pools, the successful fixed block
interval, and Swap logs (NOT all Uniswap pools/trades, ERC20 transfers or all users).
A node serving incomplete logs undetectably is outside this collector's guarantees.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests

RPC_DEFAULT = "https://ethereum-rpc.publicnode.com"
TOPIC_SWAP = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
TOKENS = {USDC: ("USDC", 6), WETH: ("WETH", 18), WBTC: ("WBTC", 8)}
# Pool contract addresses confirmed in public Uniswap pool listings / datasets.
# token0/token1 order is verified against the pool on-chain at execution time.
POOLS = {
    "usdc_weth_005": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    "usdc_weth_030": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
    "wbtc_weth_030": "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
}
COLUMNS = ["blockchain", "project", "version", "block_time", "block_number", "tx_hash",
           "evt_index", "tx_from", "tx_to", "taker", "maker", "project_contract_address",
           "token_bought_address", "token_sold_address", "token_bought_symbol", "token_sold_symbol",
           "token_bought_amount", "token_sold_amount", "amount_usd"]


def rpc(session, url, method, params):
    err = None
    for attempt in range(5):
        try:
            response = session.post(url, json={"jsonrpc": "2.0", "id": 1,
                                               "method": method, "params": params}, timeout=35)
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(str(body["error"]))
            if "result" not in body or body["result"] is None:
                raise RuntimeError(f"Missing {method} result")
            return body["result"]
        except (requests.RequestException, ValueError, RuntimeError) as e:
            err = e
            if attempt < 4:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{method} failed after retries: {err}")


def token_from_call(session, url, pool, selector, block_hex):
    data = rpc(session, url, "eth_call", [{"to": pool, "data": selector}, block_hex])
    if not isinstance(data, str) or len(data) < 66:
        raise ValueError(f"Invalid pool token response for {pool}")
    address = "0x" + data[-40:].lower()
    if address not in TOKENS:
        raise ValueError(f"Unsupported token {address} in pool {pool}; stop rather than assume decimals")
    return address


def decode_swap(log, token0, token1):
    payload = log["data"]
    if not isinstance(payload, str) or not payload.startswith("0x") or len(payload) < 130:
        raise ValueError("Invalid Swap event payload")
    def signed(word):
        number = int(word, 16)
        return number - (1 << 256) if number >= (1 << 255) else number
    a0, a1 = signed(payload[2:66]), signed(payload[66:130])
    if a0 > 0 and a1 < 0:
        sold, bought, raw_in, raw_out = token0, token1, a0, -a1
    elif a1 > 0 and a0 < 0:
        sold, bought, raw_in, raw_out = token1, token0, a1, -a0
    else:
        raise ValueError("Unexpected zero/same-direction Swap token amounts; refusing to drop a log")
    in_sym, in_dec = TOKENS[sold]
    out_sym, out_dec = TOKENS[bought]
    return sold, bought, in_sym, out_sym, str(Decimal(raw_in) / (10 ** in_dec)), str(Decimal(raw_out) / (10 ** out_dec))


def collect(args):
    with requests.Session() as session:
        chain = int(rpc(session, args.rpc, "eth_chainId", []), 16)
        if chain != 1:
            raise RuntimeError(f"Expected Ethereum mainnet chain ID 1; got {chain}")
        head = int(rpc(session, args.rpc, "eth_blockNumber", []), 16)
        last_safe = head - args.confirmations
        end = args.end_block if args.end_block is not None else last_safe
        if end > last_safe or end < 0:
            raise ValueError(f"--end-block must be <= latest confirmed block {last_safe}")
        start = end - args.blocks + 1
        if start < 0:
            raise ValueError("Invalid block interval")
        print(f"Scanning ALL Swap logs for 3 pools in blocks {start}..{end} (inclusive)", flush=True)
        pool_tokens = {}
        for name, address in POOLS.items():
            token0 = token_from_call(session, args.rpc, address, "0x0dfe1681", hex(end))
            token1 = token_from_call(session, args.rpc, address, "0xd21220a7", hex(end))
            if token0 == token1:
                raise ValueError(f"Identical tokens returned for {name}")
            pool_tokens[address] = (token0, token1)
            print(f"Verified {name}: {TOKENS[token0][0]}/{TOKENS[token1][0]}", flush=True)

        all_logs = []
        covered = 0
        for low in range(start, end + 1, args.chunk_size):
            high = min(end, low + args.chunk_size - 1)
            subset = rpc(session, args.rpc, "eth_getLogs", [{"fromBlock": hex(low),
                "toBlock": hex(high), "address": list(POOLS.values()), "topics": [TOPIC_SWAP]}])
            if not isinstance(subset, list):
                raise RuntimeError(f"Invalid logs response for {low}-{high}")
            for log in subset:
                addr = log.get("address", "").lower()
                bn = int(log["blockNumber"], 16)
                if addr not in pool_tokens or not low <= bn <= high or log["topics"][0].lower() != TOPIC_SWAP:
                    raise RuntimeError(f"Unexpected log returned in block chunk {low}-{high}")
                if log.get("removed"):
                    raise RuntimeError("Removed/reorged log encountered; rerun with a fixed safe block range")
                all_logs.append(log)
            covered += high - low + 1
            if covered % max(1, 10 * args.chunk_size) == 0 or high == end:
                print(f"Scanned {covered}/{args.blocks} blocks; {len(all_logs)} swap logs", flush=True)
            if len(all_logs) > args.max_events:
                raise RuntimeError(f"Found >{args.max_events} logs; reduce --blocks or raise --max-events. No sampling performed.")
            if args.sleep:
                time.sleep(args.sleep)

        all_logs.sort(key=lambda x: (int(x["blockNumber"], 16), int(x["logIndex"], 16)))
        keys = [(x["transactionHash"].lower(), int(x["logIndex"], 16)) for x in all_logs]
        if len(keys) != len(set(keys)):
            raise RuntimeError("Duplicate event identity in logs; refusing to save")
        print(f"Found {len(all_logs)} complete swap logs in selected pools; enriching transaction origins", flush=True)
        block_cache = {}
        tx_cache = {}
        rows = []
        counts = {name: 0 for name in POOLS}
        name_by_address = {v: k for k, v in POOLS.items()}
        for i, log in enumerate(all_logs, start=1):
            block = log["blockNumber"]
            tx_hash = log["transactionHash"]
            address = log["address"].lower()
            if block not in block_cache:
                block_cache[block] = rpc(session, args.rpc, "eth_getBlockByNumber", [block, False])
            if tx_hash not in tx_cache:
                tx_cache[tx_hash] = rpc(session, args.rpc, "eth_getTransactionByHash", [tx_hash])
            tx = tx_cache[tx_hash]
            if not tx or not tx.get("from") or not block_cache[block] or tx["hash"].lower() != tx_hash.lower():
                raise RuntimeError(f"Incomplete transaction/block metadata for {tx_hash}; no partial output written")
            sold, bought, sold_sym, bought_sym, amount_in, amount_out = decode_swap(log, *pool_tokens[address])
            timestamp = datetime.fromtimestamp(int(block_cache[block]["timestamp"], 16), tz=timezone.utc).isoformat()
            rows.append({"blockchain": "ethereum", "project": "uniswap", "version": 3,
                "block_time": timestamp, "block_number": int(block, 16), "tx_hash": tx_hash,
                "evt_index": int(log["logIndex"], 16), "tx_from": tx["from"],
                "tx_to": tx.get("to") or "", "taker": "", "maker": "",
                "project_contract_address": address,
                "token_bought_address": bought, "token_sold_address": sold,
                "token_bought_symbol": bought_sym, "token_sold_symbol": sold_sym,
                "token_bought_amount": amount_out, "token_sold_amount": amount_in, "amount_usd": ""})
            counts[name_by_address[address]] += 1
            if i % 100 == 0 or i == len(all_logs):
                print(f"Decoded {i}/{len(all_logs)} events", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
        return rows, {"source": "ethereum mainnet public JSON-RPC", "rpc": args.rpc,
            "start_block": start, "end_block": end, "confirmations_at_start": head - end,
            "blocks_scanned": covered, "pools": POOLS, "counts_per_pool": counts,
            "swap_log_count": len(all_logs), "csv_event_count": len(rows),
            "sampling": False, "usd_valuations": "not computed", "transfers": "not collected",
            "labels": "none; neither fraud nor legitimate status verified",
            "limitations": "Complete Swap logs for specified pools and interval only; tx_from not proven beneficial owner"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=120, help="consecutive Ethereum blocks, 1-1000")
    parser.add_argument("--end-block", type=int, default=None, help="optional pinned last block for reproducibility")
    parser.add_argument("--confirmations", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=5, help="blocks per eth_getLogs call (1-20)")
    parser.add_argument("--max-events", type=int, default=2000, help="fail if more logs; NEVER sample")
    parser.add_argument("--rpc", default=RPC_DEFAULT)
    parser.add_argument("--sleep", type=float, default=0.05, help="delay between public RPC calls")
    parser.add_argument("--output", default="data/raw/recent/uniswap_v3_multi_pool_complete.csv")
    args = parser.parse_args()
    if not (1 <= args.blocks <= 1000 and 1 <= args.chunk_size <= 20 and
            args.max_events >= 1 and args.confirmations >= 12 and args.sleep >= 0):
        parser.error("Invalid range: --blocks 1..1000, --chunk-size 1..20, confirmations >= 12, --max-events >= 1")
    out = Path(args.output)
    if out.exists() or out.with_suffix(".manifest.json").exists():
        parser.error(f"Output already exists: {out}. Choose a new --output to avoid overwriting a dataset.")
    rows, manifest = collect(args)
    if not rows:
        raise RuntimeError("No Swap events found; nothing saved. Try a larger --blocks value")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, out)
        manifest_path = out.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"SUCCESS: {len(rows)} unsampled swaps from {manifest['blocks_scanned']} consecutive blocks")
    print(f"CSV: {out}\nManifest: {out.with_suffix('.manifest.json')}")
    print(f"Per-pool: {manifest['counts_per_pool']}")
    print("No fraud labels, USD valuations, or transfer paths were inferred.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, requests.RequestException, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
