"""Dataset-specific, evidence-preserving adapters for historical LOB and recent AMM DEX data.

Not an exchange owner-identity resolver. Actor for AMM = transaction origin (tx_from),
not necessarily ultimate beneficiary. Maker/taker fields from AMM are NOT interpreted
as a peer-to-peer trade. Historical raw token units require token_decimals.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd

CANONICAL = ["event_id", "source", "chain", "protocol", "venue_type", "timestamp", "tx_hash", "evt_index",
             "actor_wallet", "counterparty_wallet", "pool", "token_in", "token_out", "amount_in",
             "amount_out", "amount_usd", "amounts_normalized", "role_note"]
SRC = {"idex": ("ethereum", "idex", "orderbook"),
       "etherdelta": ("ethereum", "etherdelta", "orderbook"),
       "uniswap": ("ethereum", "uniswap_v3", "amm"),
       "pancakeswap": ("bnb", "pancakeswap_v3", "amm")}


def _col(df, *names, default=None):
    for name in names:
        if name in df:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _str(v):
    return "" if pd.isna(v) else str(v).strip().lower()


def _dec(v, decimals):
    try:
        if pd.isna(v) or decimals is None:
            return np.nan
        return float(Decimal(str(v)) / Decimal(10) ** int(decimals))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return np.nan


def load_decimals(path: str | Path | None):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Token decimals metadata missing: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    mapping = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            address = _str(value.get("address", key))
            d = value.get("decimals")
        else:
            address, d = _str(key), value
        try:
            if d is not None and 0 <= int(d) <= 36:
                mapping[address] = int(d)
        except (ValueError, TypeError):
            pass
    # The native ETH sentinel in old EtherDelta is often zero address.
    mapping.setdefault("0x0000000000000000000000000000000000000000", 18)
    return mapping


def _time(values):
    # Numeric UNIX seconds/milliseconds or UTC ISO text. Prevent mixed-unit mistakes.
    txt = values.astype(str)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > .95:
        med = numeric.median()
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(txt, utc=True, errors="coerce")


def normalize(df: pd.DataFrame, source: str, decimals=None) -> pd.DataFrame:
    if source not in SRC:
        raise ValueError(f"Unknown source {source}; choose {list(SRC)}")
    if df.empty:
        return pd.DataFrame(columns=CANONICAL)
    chain, protocol, typ = SRC[source]
    decimals = decimals or {}
    # Failed executions are not successful DEX trades. Older pre-Byzantium rows
    # can have absent status: retain but do not misrepresent them as verified.
    if source == "idex" and "status" in df:
        status = df["status"].astype(str).str.lower().str.strip()
        failed = status.isin(["0", "0.0", "false", "failed", "failure"])
        df = df.loc[~failed].copy()
    if df.empty:
        return pd.DataFrame(columns=CANONICAL)
    if typ == "amm":
        mandatory = {"block_time", "tx_hash", "tx_from", "project_contract_address", "token_bought_address", "token_sold_address"}
    elif source == "idex":
        mandatory = {"timestamp", "maker", "taker", "tokenBuy", "tokenSell", "transaction_hash"} if "transactionHash" not in df else {"timestamp", "maker", "taker", "tokenBuy", "tokenSell", "transactionHash"}
    else:
        mandatory = {"timestamp", "tokenGet", "tokenGive", "get", "give", "transaction_hash"} if "transactionHash" not in df and "tokenBuy" not in df else set()
    missing = mandatory.difference(df.columns)
    if missing:
        raise ValueError(f"{source} required columns missing: {sorted(missing)}; got {list(df.columns)}")

    o = pd.DataFrame(index=df.index)
    o["source"] = source
    o["chain"] = chain
    o["protocol"] = protocol
    o["venue_type"] = typ
    o["timestamp"] = _time(_col(df, "block_time", "timestamp"))
    o["tx_hash"] = _col(df, "tx_hash", "transaction_hash", "transactionHash").map(_str)
    if typ == "amm":
        if "blockchain" in df and not (df["blockchain"].astype(str).str.lower() == chain).all():
            raise ValueError(f"Unexpected blockchain in {source} export")
        if "project" in df and not (df["project"].astype(str).str.lower() == ("pancakeswap" if source == "pancakeswap" else "uniswap")).all():
            raise ValueError(f"Unexpected DEX project in {source} export")
        if "version" in df and not (df["version"].astype(str).str.replace(".0", "", regex=False) == "3").all():
            raise ValueError("Expected V3 swaps only")
        o["evt_index"] = pd.to_numeric(_col(df, "evt_index"), errors="coerce")
        o["actor_wallet"] = _col(df, "tx_from").map(_str)
        o["counterparty_wallet"] = ""
        o["pool"] = _col(df, "project_contract_address").map(_str)
        o["token_in"] = _col(df, "token_sold_address").map(_str)
        o["token_out"] = _col(df, "token_bought_address").map(_str)
        o["amount_in"] = pd.to_numeric(_col(df, "token_sold_amount"), errors="coerce")
        o["amount_out"] = pd.to_numeric(_col(df, "token_bought_amount"), errors="coerce")
        o["amount_usd"] = pd.to_numeric(_col(df, "amount_usd"), errors="coerce")
        o["amounts_normalized"] = True
        o["role_note"] = "tx_from = tx origin; not guaranteed end-beneficiary; pool != wallet counterparty"
    else:
        o["evt_index"] = pd.to_numeric(_col(df, "evt_index"), errors="coerce")
        o["actor_wallet"] = _col(df, "taker", "give").map(_str)
        o["counterparty_wallet"] = _col(df, "maker", "get").map(_str)
        o["pool"] = ""
        o["token_in"] = _col(df, "tokenSell", "tokenGive").map(_str)
        o["token_out"] = _col(df, "tokenBuy", "tokenGet").map(_str)
        # On IDEX the tokenBuy/tokenSell amounts are defined by the maker order,
        # NOT a proven inflow/outflow of the taker. Do not infer net balances.
        preprocessed = ("amountBoughtReal" in df and "amountSoldReal" in df)
        if preprocessed:
            o["amount_in"] = pd.to_numeric(df["amountSoldReal"], errors="coerce")
            o["amount_out"] = pd.to_numeric(df["amountBoughtReal"], errors="coerce")
        elif source == "etherdelta":
            o["amount_in"] = [_dec(v, decimals.get(_str(t))) for v, t in zip(_col(df, "amountGive"), o["token_in"])]
            o["amount_out"] = [_dec(v, decimals.get(_str(t))) for v, t in zip(_col(df, "amountGet"), o["token_out"])]
        else:
            # IDEX raw amount = actual filled tokenBuy quantity; amountSell is ORDER amount,
            # not executed sold amount. Derive the fill ratio from raw order quantities.
            buy_raw = _col(df, "amountBuy")
            sell_raw = _col(df, "amountSell")
            filled = _col(df, "amount")
            o["amount_out"] = [_dec(v, decimals.get(_str(t))) for v, t in zip(filled, o["token_out"])]
            def sold_actual(a, b, fill, token):
                try:
                    if decimals.get(_str(token)) is None or Decimal(str(a)) == 0:
                        return np.nan
                    raw = Decimal(str(fill)) * Decimal(str(b)) / Decimal(str(a))
                    return float(raw / Decimal(10) ** decimals[_str(token)])
                except (InvalidOperation, ValueError, TypeError, ZeroDivisionError):
                    return np.nan
            o["amount_in"] = [sold_actual(a, b, fill, t) for a,b,fill,t in zip(buy_raw, sell_raw, filled, o["token_in"])]
        o["amount_usd"] = np.nan  # price data needs verified external join
        o["amounts_normalized"] = preprocessed or (o["amount_in"].notna() & o["amount_out"].notna())
        o["role_note"] = "orderbook maker/taker observed; token direction follows maker order, not inferred taker balance"
    if o["timestamp"].isna().any():
        raise ValueError(f"{source}: invalid timestamp rows present")
    if o["tx_hash"].eq("").any():
        raise ValueError(f"{source}: missing transaction hash")
    # For historical same tx_hash can contain several events. Stable occurrence index within
    # provided sorted export; preserve original row id to audit ambiguity.
    fallback_idx = o.groupby("tx_hash", sort=False).cumcount()
    idx = o["evt_index"].fillna(fallback_idx).astype("int64")
    o["evt_index"] = idx
    o["event_id"] = o["chain"] + ":" + o["protocol"] + ":" + o["tx_hash"] + ":" + idx.astype(str)
    for col in ["actor_wallet", "token_in", "token_out"]:
        if o[col].eq("").any():
            raise ValueError(f"{source}: empty {col} in raw data. Audit wallet roles before processing.")
    if o["event_id"].duplicated().any():
        raise ValueError(f"{source}: duplicate event IDs (tx hash/event index collision)")
    if ((o["amount_in"].dropna() < 0).any() or (o["amount_out"].dropna() < 0).any()):
        raise ValueError(f"{source}: negative normalized amounts")
    return o[CANONICAL].sort_values(["timestamp", "event_id"]).reset_index(drop=True)


def normalize_transfers(df: pd.DataFrame, source_chain: str):
    required = {"block_time", "tx_hash", "from_address", "to_address"}
    if required.difference(df.columns):
        raise ValueError(f"Transfer export missing {required.difference(df.columns)}")
    t = pd.DataFrame()
    t["timestamp"] = _time(df["block_time"])
    t["chain"] = source_chain
    if "blockchain" in df and not df["blockchain"].astype(str).str.lower().eq(source_chain).all():
        raise ValueError("Chain mismatch in transfers")
    t["tx_hash"] = df["tx_hash"].map(_str)
    t["from_wallet"] = df["from_address"].map(_str)
    t["to_wallet"] = df["to_address"].map(_str)
    t["token"] = _col(df, "token_address", "contract_address").map(_str)
    t["amount"] = pd.to_numeric(_col(df, "amount"), errors="coerce")
    t["evt_index"] = pd.to_numeric(_col(df, "evt_index"), errors="coerce").fillna(0).astype(int)
    t["transfer_id"] = t["chain"] + ":" + t["tx_hash"] + ":" + t["evt_index"].astype(str)
    if t["timestamp"].isna().any() or t[["from_wallet", "to_wallet"]].eq("").any().any():
        raise ValueError("Transfer export has invalid timestamps or missing addresses")
    return t.drop_duplicates("transfer_id").sort_values("timestamp").reset_index(drop=True)


def _read_limited(path, n):
    # Don't read >3 GB historical datasets fully into memory.
    return pd.read_csv(path, nrows=n if n else None, low_memory=False)


def dataset_inventory(paths):
    inventory = {}
    for name, path in paths.items():
        if path:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(p)
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            inventory[name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}
    return inventory


def run_import(args):
    sources = {s: getattr(args, s) for s in SRC}
    valid = {k: v for k, v in sources.items() if v}
    if not valid:
        raise ValueError("No actual input CSVs supplied. See README for --demo-synthetic.")
    decimals = load_decimals(args.decimals)
    frames = [normalize(_read_limited(path, args.limit_per_source), s, decimals) for s, path in valid.items()]
    trades = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    trans_frames = []
    for ch, attr in [("ethereum", "ethereum_transfers"), ("bnb", "bnb_transfers")]:
        path = getattr(args, attr)
        if path:
            trans_frames.append(normalize_transfers(_read_limited(path, args.limit_per_source), ch))
    transfers = pd.concat(trans_frames, ignore_index=True) if trans_frames else pd.DataFrame(columns=["transfer_id","timestamp","chain","tx_hash","from_wallet","to_wallet","token","amount","evt_index"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "trades.csv", index=False)
    transfers.to_csv(output / "transfers.csv", index=False)
    summary = {"sources": trades.source.value_counts().to_dict(), "transfers": len(transfers),
               "missing_usd_fraction": float(trades.amount_usd.isna().mean()),
               "missing_normalized_amount_fraction": float(trades.amount_in.isna().mean()),
               "first_event": str(trades.timestamp.min()), "last_event": str(trades.timestamp.max()),
               "provenance": dataset_inventory({**valid, "decimals": args.decimals,
                         "ethereum_transfers":args.ethereum_transfers,"bnb_transfers":args.bnb_transfers}),
               "warning": "No confirmed organic/fraud labels in these raw DEX exports."}
    (output / "data_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return trades, transfers


def main():
    p = argparse.ArgumentParser(description="Normalize dataset-specific real DEX CSVs")
    for s in SRC:
        p.add_argument("--" + s)
    p.add_argument("--ethereum-transfers")
    p.add_argument("--bnb-transfers")
    p.add_argument("--decimals")
    p.add_argument("--limit-per-source", type=int, default=50000,
                   help="Max first N rows per source for pilot; increase only after validating schema")
    p.add_argument("--output", default="data/processed")
    run_import(p.parse_args())

if __name__ == "__main__":
    main()
