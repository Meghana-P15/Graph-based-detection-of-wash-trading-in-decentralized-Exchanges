"""Data, privacy, label and human-review guardrails for the research prototype.

No automatic banning, ownership inference, or real-world person identification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .features import load_processed


def safety_report(processed='data/processed',reports='reports',source='uniswap'):
    trades,transfers=load_processed(processed)
    trades=trades[trades.source==source]
    if not len(trades):
        raise ValueError('Dataset empty: '+source)
    r=Path(reports)/source
    preds=r/'inference_predictions.csv'
    if not preds.exists():
        raise FileNotFoundError('Run src.infer first: '+str(preds))
    p=pd.read_csv(preds)
    # Conservative review queue: ambiguity and high-risk predictions require human review.
    queue=p[(p.review_required.astype(str).str.lower()=='true') |
            (p.candidate_decision=='suspicious_candidate')].copy()
    queue['investigator_disposition']='NOT_REVIEWED'
    queue['reviewer_notes']=''
    queue['confirmed_fraud']='UNDETERMINED'
    queue['ownership_inference_prohibited']=True
    queue.to_csv(r/'human_review_queue.csv',index=False)
    report={
        'scope':'research / public pseudonymous DEX trade investigation',
        'source':source,
        'imported_source_events':int(len(trades)),
        'scored_events':int(len(p)),
        'duplicate_event_ids':int(trades.event_id.duplicated().sum()),
        'missing_timestamps':int(trades.timestamp.isna().sum()),
        'missing_actor_wallets':int(trades.actor_wallet.isna().sum()),
        'missing_usd_amount_fraction':float(trades.amount_usd.isna().mean()),
        'unreviewed_queue_size':int(len(queue)),
        'offchain_identity_claims':False,
        'automatic_wallet_blocking':False,
        'verified_fraud_label_count':0,
        'label_interpretation':'Candidate heuristic labels, not adjudicated intent or organic status.',
        'important_caveats':['AMM tx origin may be a router/keeper, not end customer',
                              'Pool co-use does not imply two wallet owners are related',
                              'Only observed transfers are used as wallet-to-wallet edges',
                              'Historical order-book tokens require correct decimal normalization',
                              'Synthetic demonstration data must never be presented as real'],
    }
    (r/'safety_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--processed',default='data/processed')
    p.add_argument('--reports',default='reports');p.add_argument('--source',default='uniswap')
    a=p.parse_args();safety_report(a.processed,a.reports,a.source)
