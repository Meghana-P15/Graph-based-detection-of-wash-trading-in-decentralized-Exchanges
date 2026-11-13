"""Observed-edge-only multi-wallet coordination analysis with audit-friendly evidence."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import networkx as nx


def analyze_groups(trades: pd.DataFrame, features: pd.DataFrame, transfers: pd.DataFrame | None=None, predictions=None):
    t=trades.copy();f=features.set_index("event_id")
    pred = predictions.set_index("event_id") if predictions is not None and len(predictions) else None
    out=[]
    for (chain,source),part in t.groupby(["chain","source"]):
        g=nx.Graph()
        for r in part.itertuples(index=False):
            a=f"{chain}:{r.actor_wallet}"; g.add_node(a)
            if r.venue_type=='orderbook' and isinstance(r.counterparty_wallet,str) and r.counterparty_wallet:
                b=f"{chain}:{r.counterparty_wallet}"; g.add_edge(a,b,kind="observed_maker_taker",event_id=r.event_id)
        if transfers is not None and len(transfers):
            relevant = transfers[(transfers.chain==chain) &
                 (transfers.timestamp<=part.timestamp.max())]
            known_pools=set(part.pool.dropna().astype(str))-{""}
            for r in relevant.itertuples(index=False):
                if r.from_wallet not in known_pools and r.to_wallet not in known_pools:
                    g.add_edge(f"{chain}:{r.from_wallet}", f"{chain}:{r.to_wallet}",
                               kind="observed_transfer",event_id=r.transfer_id)
        for i,component in enumerate(nx.connected_components(g)):
            if len(component)<2:
                continue
            addresses={x.split(':',1)[1] for x in component}
            events=part[(part.actor_wallet.isin(addresses)) |
                        (part.counterparty_wallet.isin(addresses) & (part.venue_type=='orderbook'))]
            risk_vals=[]
            for eid in events.event_id:
                if pred is not None and eid in pred.index:
                    risk_vals.append(float(pred.loc[eid,"risk_score"]))
            weak_hits=int(f.loc[events.event_id,"weak_label"].eq(1).sum())
            graph=g.subgraph(component)
            evidence=[{"from":a,"to":b,"relation":info.get('kind'),
                       "event_id":info.get('event_id')} for a,b,info in graph.edges(data=True)]
            out.append(dict(group_id=f"{chain}:{source}:{i}",chain=chain,source=source,
                 wallet_count=len(addresses),trade_event_count=len(events),
                 weak_suspicious_count=weak_hits,
                 model_risk_mean=round(sum(risk_vals)/len(risk_vals),4) if risk_vals else None,
                 evidence=json.dumps(evidence[:100]),
                 limitations="Observed connections do not establish common beneficial ownership; investigate benign routes/arbitrage."))
        # AMM: connected components mostly isolated without transfer edges. Add a CO-ACTIVITY candidate
        # only if at least two wallets traded same pool in same 5-minute bucket repeatedly.
        amm=part[part.venue_type=='amm'].copy()
        if len(amm):
            amm['bucket']=amm.timestamp.dt.floor('5min')
            correlations={}
            for (pool,bucket), grp in amm.groupby(['pool','bucket']):
                wallets=sorted(set(grp.actor_wallet))
                if len(wallets)>=2:
                    pairkey=(pool,tuple(wallets)); correlations[pairkey]=correlations.get(pairkey,0)+1
            for (pool,wallets),repeat in correlations.items():
                if repeat<2: continue
                events=amm[(amm.pool==pool)&(amm.actor_wallet.isin(wallets))]
                out.append(dict(group_id=f"{chain}:{source}:coactivity:{pool[-6:]}:{wallets[0][-5:]}",
                     chain=chain,source=source,wallet_count=len(wallets),trade_event_count=len(events),
                     weak_suspicious_count=int(f.loc[events.event_id,"weak_label"].eq(1).sum()),
                     model_risk_mean=None, evidence=json.dumps([{"pool":pool,"wallets":wallets,"repeated_5m_buckets":repeat}]),
                     limitations="Pool co-activity is a correlation candidate, NOT a wallet transfer or proof of common owner."))
    return pd.DataFrame(out, columns=["group_id","chain","source","wallet_count","trade_event_count",
                 "weak_suspicious_count","model_risk_mean","evidence","limitations"])
