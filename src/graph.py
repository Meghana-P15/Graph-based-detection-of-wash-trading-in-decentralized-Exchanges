"""Leakage-safe event ego-graphs: build each target's graph from current + PRIOR DEX events.

GCN/GraphSAGE operate over the typed entity projection (one-hot node types).
They are NOT presented as full relational/temporal heterogeneous neural networks.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

NODE_DIM = 12
NODE_TYPES = {"event":0, "wallet":1, "pool":2, "token":3}


@dataclass
class EventGraph:
    event_id: str
    timestamp: pd.Timestamp
    source: str
    label: int
    x: torch.Tensor
    edge_index: torch.Tensor
    nodes: list
    links: list


def _nodefeatures(typ, amount=0., observed=0., is_root=False, is_amm=False):
    # No raw weak-label rule outputs are directly injected as input features.
    v = np.zeros(NODE_DIM, dtype=np.float32)
    v[NODE_TYPES[typ]] = 1
    v[4] = float(np.log1p(max(float(amount), 0))) if np.isfinite(amount) else 0.
    v[5] = float(np.log1p(max(observed, 0)))
    v[6] = float(is_root)
    v[7] = float(is_amm)
    return v


def _safe_str(v):
    return "" if pd.isna(v) else str(v)


def make_graph_for_event(row, history: pd.DataFrame, label=-1, transfer_paths=None, max_past=25):
    """row = normalized DEX trade. history must be same DEX/chain and strictly older."""
    t = row.timestamp
    if len(history) and not (pd.to_datetime(history.timestamp, utc=True) < t).all():
        raise ValueError("Leakage: historical ego-graph received current/future events")
    nodes = []
    mapping = {}
    edge_pairs = []
    links = []
    source_venue = row.venue_type

    def add(typ, key, amount=0., observed=0., root=False):
        key = f"{typ}:{row.chain}:{key}"
        if key not in mapping:
            mapping[key] = len(nodes)
            nodes.append({"id":key,"type":typ,
                          "features":_nodefeatures(typ, amount, observed, root, source_venue=="amm")})
        return mapping[key]

    def connect(a,b,rel):
        edge_pairs.extend([(a,b),(b,a)])
        links.append({"source":nodes[a]["id"],"target":nodes[b]["id"],"relation":rel})

    amount = float(row.amount_usd) if pd.notna(row.amount_usd) and row.amount_usd >= 0 else 0
    center=add("event", row.event_id, amount, root=True)
    actor=add("wallet", row.actor_wallet, observed=int((history.actor_wallet == row.actor_wallet).sum()))
    connect(center,actor,"event_actor_tx_origin" if source_venue=="amm" else "event_taker")
    if source_venue == "orderbook" and _safe_str(row.counterparty_wallet):
        maker=add("wallet", row.counterparty_wallet, observed=int((history.counterparty_wallet==row.counterparty_wallet).sum()))
        connect(center,maker,"event_observed_maker")
    if source_venue == "amm" and _safe_str(row.pool):
        pool=add("pool", row.pool, observed=int((history.pool == row.pool).sum()))
        connect(center,pool,"event_uses_pool")
    for col,rel in [("token_in","event_sells_token"),("token_out","event_buys_token")]:
        val=_safe_str(getattr(row,col))
        if val:
            tok=add("token",val)
            connect(center,tok,rel)

    # Past activity is connected VIA OBSERVED past event entities, never fictitious wallet-to-wallet trade edges.
    prev = history[(history.actor_wallet==row.actor_wallet) |
                   ((row.pool != "") & (history.pool == row.pool)) |
                   ((row.counterparty_wallet != "") & (history.counterparty_wallet == row.counterparty_wallet))]
    prev = prev.sort_values("timestamp").tail(max_past)
    for p in prev.itertuples(index=False):
        a = float(p.amount_usd) if pd.notna(p.amount_usd) and p.amount_usd>=0 else 0.
        pe=add("event",p.event_id,a)
        pw=add("wallet",p.actor_wallet)
        connect(pe,pw,"prior_event_actor")
        if p.venue_type=="orderbook" and _safe_str(p.counterparty_wallet):
            maker=add("wallet",p.counterparty_wallet)
            connect(pe,maker,"prior_event_maker")
        if p.venue_type=="amm" and _safe_str(p.pool):
            po=add("pool",p.pool)
            connect(pe,po,"prior_event_pool")
    # Transfer paths precomputed strictly before event, provenance preserved.
    for evidence in (transfer_paths or [])[:3]:
        wallets = evidence.get("wallets",[])
        transfer_ids = evidence.get("transfer_ids",[])
        for k in range(len(wallets)-1):
            n1 = add("wallet", wallets[k]); n2 = add("wallet", wallets[k+1])
            connect(n1,n2,"observed_transfer:"+(transfer_ids[k] if k<len(transfer_ids) else "unknown"))
    if not edge_pairs:
        edge_pairs=[(center,center)]
    x=torch.tensor(np.stack([n["features"] for n in nodes]),dtype=torch.float32)
    ei=torch.tensor(edge_pairs,dtype=torch.long).T.contiguous()
    for n in nodes:
        n["features"] = n["features"].tolist()
    return EventGraph(row.event_id,row.timestamp,row.source,int(label),x,ei,nodes,links)


def build_graphs(trades, features, max_past=25, lookback_h=24, max_events=None):
    # Causal, source-specific histories; global chronological iteration avoids future-data leakage.
    t = trades.copy().sort_values(["timestamp","event_id"]).reset_index(drop=True)
    t["timestamp"] = pd.to_datetime(t.timestamp,utc=True)
    f=features.set_index("event_id")
    seen = defaultdict(list)
    output = []
    for n,row in enumerate(t.itertuples(index=False)):
        if max_events is not None and n >= max_events:
            break
        scope=(row.chain,row.source)
        history = pd.DataFrame(seen[scope],columns=t.columns)
        if len(history):
            history["timestamp"] = pd.to_datetime(history.timestamp, utc=True)
            history=history[(history.timestamp>=row.timestamp-pd.Timedelta(hours=lookback_h)) &
                            (history.timestamp<row.timestamp)]
        if row.event_id not in f.index:
            raise ValueError("Missing event features: " + row.event_id)
        feat=f.loc[row.event_id]
        paths=json.loads(feat.get("transfer_paths", "[]"))
        output.append(make_graph_for_event(row,history,int(feat.weak_label),paths,max_past))
        seen[scope].append(tuple(row))
    return output


def batch_graphs(graphs):
    """Create a disjoint union and return indices of each ROOT event node."""
    xs=[]; eis=[]; rootidx=[]; ys=[]; offset=0
    for g in graphs:
        xs.append(g.x)
        eis.append(g.edge_index+offset)
        rootidx.append(offset)  # root always first
        ys.append(g.label)
        offset += len(g.x)
    return (torch.cat(xs,0), torch.cat(eis,1),
            torch.tensor(rootidx,dtype=torch.long),torch.tensor(ys,dtype=torch.float32))
