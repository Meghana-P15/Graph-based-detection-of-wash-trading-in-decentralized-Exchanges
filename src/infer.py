"""Score selected DEX source with a SAVED model; outputs investigatory candidate classifications."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .features import prepare_features,load_processed
from .graph import build_graphs,batch_graphs
from .model import EventGNN


def run_inference(processed='data/processed',source='uniswap',arch='sage',modeldir='models',outdir='reports',max_events=1200):
    trades,transfers=load_processed(processed)
    trades=trades.loc[trades.source==source].sort_values('timestamp').head(max_events).reset_index(drop=True)
    if not len(trades):
        raise ValueError('No trades for ' + source)
    if len(transfers):
        transfers=transfers.loc[transfers.chain.isin(trades.chain.unique())].copy()
    feats=prepare_features(trades,transfers)
    root=Path(modeldir)/source
    meta=json.loads((Path(outdir)/source/'metrics.json').read_text())
    model_metrics=meta['models'].get(arch)
    if not model_metrics:
        raise ValueError(f'No trained model {arch} for {source}; run src.train first')
    threshold=float(model_metrics['threshold'])
    if arch in ('sage','gcn'):
        checkpoint=torch.load(root/arch/'model.pt',map_location='cpu',weights_only=True)
        model=EventGNN(arch,checkpoint['input_dim'],checkpoint['hidden_dim'])
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        graphs=build_graphs(trades,feats,max_events=max_events)
        x,e,r,_=batch_graphs(graphs)
        with torch.no_grad():
            scores=torch.sigmoid(model(x,e,r)[0]).numpy()
    else:
        artifact=joblib.load(root/(arch+'.joblib'))
        scores=artifact['model'].predict_proba(feats[artifact['features']].fillna(0))[:,1]
        threshold=float(artifact['threshold'])
    # Abstain on ambiguous weak-label evidential support or score close to threshold;
    # label is NOT a real fraud ground-truth category.
    abstain=(feats.weak_label==-1) | (np.abs(scores-threshold)<.08)
    decisions=np.where(abstain,'review_required',np.where(scores>=threshold,'suspicious_candidate','organic_candidate'))
    out=feats[['event_id','timestamp','source','actor_wallet','weak_label','label_reason_codes','transfer_paths']].copy()
    out['risk_score']=scores
    out['candidate_decision']=decisions
    out['review_required']=abstain
    out['model']=arch
    out['classification_scope']='investigation_candidate_not_proven_fraud'
    dest=Path(outdir)/source;dest.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest/'inference_predictions.csv',index=False)
    print(f'Saved {len(out)} scored events from {source} to {dest}/inference_predictions.csv')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed',default='data/processed');p.add_argument('--source',default='uniswap')
    p.add_argument('--arch',default='sage',choices=['sage','gcn','logistic_regression','random_forest'])
    p.add_argument('--modeldir',default='models');p.add_argument('--outdir',default='reports')
    p.add_argument('--max-events',type=int,default=1200)
    a=p.parse_args()
    run_inference(a.processed,a.source,a.arch,a.modeldir,a.outdir,a.max_events)
