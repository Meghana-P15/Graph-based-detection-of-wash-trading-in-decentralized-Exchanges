"""Past-only transaction features and transparent weak candidate labels.

The labels are HEURISTIC AGREEMENT, not independently verified fraud labels.
Each event is evaluated using its own current record and past events only.
"""
from __future__ import annotations
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

NUMERIC = ["log_amount_usd", "has_amount_usd", "prior_actor_1h", "prior_pair_1h",
           "prior_pool_1h", "prior_reverse_1h", "prior_counterparty_1h",
           "prior_other_wallets_pool_5m", "prior_funding_hops", "hour_sin", "hour_cos",
           "is_amm", "missing_normalized_amount"]
# Advised leak-sensitivity evaluation excludes the exact heuristic features in label rule.
STRICT = ["log_amount_usd", "has_amount_usd", "hour_sin", "hour_cos", "is_amm",
          "missing_normalized_amount"]


def find_prior_transfer_paths(transfers, actor, chain, cutoff, forbidden=None, max_hops=4, window_hours=24):
    """Paths ending at actor with strictly increasing transfer timestamps; no inferred ownership."""
    forbidden = forbidden or set()
    if not actor or not len(transfers):
        return []
    t = transfers[(transfers.chain == chain) & (transfers.timestamp < cutoff) &
                  (transfers.timestamp >= cutoff - pd.Timedelta(hours=window_hours))]
    t = t[(~t.from_wallet.isin(forbidden)) & (~t.to_wallet.isin(forbidden))]
    backwards = defaultdict(list)
    for r in t.itertuples(index=False):
        if r.from_wallet != r.to_wallet:
            backwards[r.to_wallet].append((r.from_wallet, r.timestamp, r.transfer_id))
    # Reverse search; require each earlier hop earlier than the next one.
    q = deque([(actor, cutoff, [actor], [])]); results = []
    while q:
        curr, latest, path, txids = q.popleft()
        if len(path) - 1 >= max_hops:
            continue
        for source, when, tid in backwards.get(curr, []):
            if when >= latest or source in path:
                continue
            new_path = [source] + path
            ids = [tid] + txids
            results.append({"wallets": new_path, "transfer_ids": ids,
                            "hops": len(new_path)-1})
            q.append((source, when, new_path, ids))
            if len(results) >= 40:
                return results
    return results


def prepare_features(trades: pd.DataFrame, transfers: pd.DataFrame | None = None, window_hours=1):
    required = {"event_id","source","chain","timestamp","actor_wallet","venue_type","pool","token_in","token_out"}
    if not required.issubset(trades.columns):
        raise ValueError(f"Normalized trades missing {required.difference(trades.columns)}")
    t = trades.copy().sort_values(["timestamp","event_id"]).reset_index(drop=True)
    t["timestamp"] = pd.to_datetime(t.timestamp, utc=True)
    transfers = transfers.copy() if transfers is not None else pd.DataFrame()
    if len(transfers):
        transfers["timestamp"] = pd.to_datetime(transfers.timestamp, utc=True)
    # Scoped to source AND chain; never combine different venue semantics or chains.
    recent = defaultdict(deque)
    results = []
    seen_start = {}
    all_pools = set(t.pool.dropna().astype(str)) - {""}
    for r in t.itertuples(index=False):
        timestamp = r.timestamp
        scope = (r.chain, r.source)
        actor = (scope, r.actor_wallet)
        pair = tuple(sorted([r.token_in, r.token_out]))
        pairkey = (scope, r.actor_wallet, pair)
        poolkey = (scope, r.pool) if r.pool else None
        cpkey = (scope, r.actor_wallet, r.counterparty_wallet) if r.counterparty_wallet else None
        reversekey = (scope, r.actor_wallet, r.token_out, r.token_in)
        keys = [actor, pairkey, reversekey]
        if poolkey: keys.append(poolkey)
        if cpkey: keys.append(cpkey)
        # Prune relevant windows. Each queue contains only PRIOR processed events.
        def history(key, minutes):
            dq = recent[key]
            threshold = timestamp - pd.Timedelta(minutes=minutes)
            while dq and dq[0][0] < threshold:
                dq.popleft()
            return [x for x in dq if x[0] < timestamp]  # no same-timestamp lookahead
        act = history(actor, window_hours*60)
        same_pair = history(pairkey, window_hours*60)
        same_pool = history(poolkey, window_hours*60) if poolkey else []
        prior_cp = history(cpkey, window_hours*60) if cpkey else []
        reverse = history(reversekey, window_hours*60)
        near_pool = [e for e in same_pool if e[0] >= timestamp - pd.Timedelta(minutes=5)]
        other_wallets = {ev[1] for ev in near_pool if ev[1] != r.actor_wallet}
        # Strict: do not treat pool/router as user-wallet funding link.
        forbidden = all_pools
        # Graph transfer evidence is computed in graph module only when supplied.
        # Keep costly multi-hop search cached at the event level for modest pilot sizes.
        paths = find_prior_transfer_paths(transfers, r.actor_wallet, r.chain, timestamp, forbidden) if len(transfers) else []
        hops = max((p["hops"] for p in paths), default=0)
        # This label policy has two independent families: trading repetition + coordination/roundtrip.
        # The explicit reasons are retained. Single symptoms remain ambiguous.
        reasons = []
        if len(act) >= 3:
            reasons.append("activity_burst_1h")
        if len(same_pair) >= 2:
            reasons.append("repeated_pair_1h")
        if len(reverse) >= 1:
            reasons.append("reverse_direction_1h")
        if len(prior_cp) >= 2:
            reasons.append("repeated_observed_counterparty_1h")
        if len(other_wallets) >= 2 and len(same_pool) >= 3:
            reasons.append("same_pool_multiwallet_5m")
        # Transfer paths alone are never evidence of common ownership or intent.
        # At least two trading-pattern indicators required for weak suspicious seed.
        independent = {
            "frequency": any(x in reasons for x in ["activity_burst_1h", "repeated_pair_1h"]),
            "roundtrip": "reverse_direction_1h" in reasons,
            "direct_network": "repeated_observed_counterparty_1h" in reasons,
            "temporal_pool": "same_pool_multiwallet_5m" in reasons,
        }
        families = sum(independent.values())
        if families >= 2:
            label = 1
        elif families == 0:
            label = 0
        else:
            label = -1
        seen_start.setdefault(scope, timestamp)
        observation_age = (timestamp - seen_start[scope]).total_seconds() / 3600
        if observation_age < 1 or not (r.actor_wallet and r.token_in and r.token_out):
            label = -1
            reasons.append("insufficient_history_or_fields")
        amount = float(r.amount_usd) if pd.notna(r.amount_usd) and r.amount_usd >= 0 else 0.
        hour = timestamp.hour + timestamp.minute/60
        results.append({"event_id":r.event_id, "timestamp":timestamp, "source":r.source,
           "chain":r.chain,"actor_wallet":r.actor_wallet,
           "log_amount_usd":np.log1p(amount),"has_amount_usd":int(pd.notna(r.amount_usd)),
           "prior_actor_1h":len(act), "prior_pair_1h":len(same_pair),
           "prior_pool_1h":len(same_pool), "prior_reverse_1h":len(reverse),
           "prior_counterparty_1h":len(prior_cp),
           "prior_other_wallets_pool_5m":len(other_wallets), "prior_funding_hops":hops,
           "hour_sin":np.sin(2*np.pi*hour/24), "hour_cos":np.cos(2*np.pi*hour/24),
           "is_amm":int(r.venue_type == "amm"),
           "missing_normalized_amount":int(not bool(r.amounts_normalized)),
           "weak_label":label, "weak_label_source":"heuristic_v1_past_only",
           "label_reason_codes":json.dumps(reasons),
           "transfer_paths":json.dumps(paths[:5]), "label_confidence":"candidate_not_verified"})
        # append after feature extraction; store time, wallet, ID as provenance
        for key in set(keys):
            recent[key].append((timestamp, r.actor_wallet, r.event_id))
        # reverse history lookup: per event store ordered token pair under actual direction key
        directkey = (scope, r.actor_wallet, r.token_in, r.token_out)
        if directkey not in set(keys):
            recent[directkey].append((timestamp, r.actor_wallet, r.event_id))
    out = pd.DataFrame(results)
    if out.event_id.duplicated().any():
        raise ValueError("Duplicate feature event ID")
    return out


def load_processed(root="data/processed"):
    root = Path(root)
    trades = pd.read_csv(root / "trades.csv", parse_dates=["timestamp"])
    transferfile = root / "transfers.csv"
    trans = pd.read_csv(transferfile, parse_dates=["timestamp"]) if transferfile.exists() and transferfile.stat().st_size>100 else pd.DataFrame()
    for c in ["pool", "counterparty_wallet"]:
        trades[c] = trades[c].fillna("")
    return trades, trans
