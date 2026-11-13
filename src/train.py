"""Train baseline + actual graph-message-passing GCN/GraphSAGE on a frozen source/time split.

All quality scores measure candidate weak-label agreement, NOT proven fraud.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
     f1_score,confusion_matrix,precision_recall_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import NUMERIC,STRICT,prepare_features,load_processed
from .graph import batch_graphs,build_graphs,NODE_DIM
from .model import EventGNN
from .coordination import analyze_groups


def chrono_split(features:pd.DataFrame):
    """Split by event chronological order first, then remove ambiguous labels from scoring."""
    f=features.sort_values(["timestamp","event_id"]).reset_index(drop=True)
    n=len(f)
    if n<25: raise ValueError("Need >=25 observed events for split; use larger real sample")
    a,b=int(.60*n),int(.80*n)
    splits={"train":f.iloc[:a].copy(),"validation":f.iloc[a:b].copy(),"test":f.iloc[b:].copy()}
    for name,s in splits.items():
        if not s.weak_label.isin([0,1]).any():
            raise ValueError(f"{name} has no confidently weak-labeled events; increase input or revise rules")
        if name=="train" and len(set(s.loc[s.weak_label>=0,"weak_label"]))<2:
            raise ValueError("Training fold requires both weak-label classes; increase sample/time window")
    return splits


def _threshold(y, scores):
    options=np.linspace(.2,.8,25)
    return float(max(options,key=lambda t:f1_score(y,np.asarray(scores)>=t,zero_division=0)))


def metric_report(y,s,threshold):
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    preds=(s>=threshold).astype(int)
    return {"n":int(len(y)),"weak_positive_count":int(y.sum()),"positive_prevalence":float(y.mean()),
            "precision":float(precision_score(y,preds,zero_division=0)),
            "recall":float(recall_score(y,preds,zero_division=0)),
            "f1":float(f1_score(y,preds,zero_division=0)),
            "macro_f1":float(f1_score(y,preds,average="macro",zero_division=0)),
            "pr_auc":float(average_precision_score(y,s)) if len(set(y))>1 else None,
            "threshold":threshold, "confusion_matrix":confusion_matrix(y,preds,labels=[0,1]).tolist()}


def train_graph_model(name, graph_by_id, splits, out, epochs=15, seed=7):
    torch.manual_seed(seed)
    model=EventGNN(arch=name,input_dim=NODE_DIM,hidden_dim=48)
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-4)
    batches={}
    for phase in splits:
        rows=splits[phase]
        selected=[graph_by_id[e] for e in rows.event_id if e in graph_by_id and graph_by_id[e].label>=0]
        if not selected:
            raise ValueError("No usable graph rows in "+phase)
        batches[phase]=(selected,batch_graphs(selected))
    train_ys=batches["train"][1][3]
    pos=float(train_ys.sum());neg=len(train_ys)-pos
    weight=torch.tensor([neg/max(pos,1)],dtype=torch.float32)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    best_loss=float('inf');best_state=None;patience=5;no_improvement=0
    started=time.perf_counter()
    for epoch in range(epochs):
        model.train();x,edge,roots,y=batches['train'][1]
        opt.zero_grad();logits,_=model(x,edge,roots)
        loss=loss_fn(logits,y);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            x,edge,roots,y=batches['validation'][1]
            logits,_=model(x,edge,roots);vl=loss_fn(logits,y).item()
        if vl<best_loss-1e-4:
            best_loss=vl;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};no_improvement=0
        else:
            no_improvement+=1
            if no_improvement>=patience: break
    model.load_state_dict(best_state);model.eval()
    savedir=out/name;savedir.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"arch":name,"input_dim":NODE_DIM,
                "hidden_dim":48,"dataset_type":"weak_labels_not_verified_fraud"},savedir/'model.pt')
    results={}
    with torch.no_grad():
        val=batches['validation'][1]
        val_prob=torch.sigmoid(model(*val[:3])[0]).numpy()
        th=_threshold(val[3].numpy(),val_prob)
        for phase,(_,batch) in batches.items():
            logits,emb=model(*batch[:3]);s=torch.sigmoid(logits).numpy()
            results[phase]=(batch[3].numpy().astype(int),s,emb.numpy())
    scores={phase:metric_report(y,s,th) for phase,(y,s,_) in results.items()}
    scores["training_seconds"]=round(time.perf_counter()-started,3)
    # Embeddings produced from actual graph neural message passing, keyed by event ID.
    with torch.no_grad():
        ge=[graph_by_id[e] for e in graph_by_id]
        all_batch=batch_graphs(ge)
        logits,emb=model(*all_batch[:3])
        node_embeddings=model.embed_nodes(all_batch[0],all_batch[1]).numpy()
        embeddings=pd.DataFrame(emb.numpy(),columns=[f"embedding_{k}" for k in range(emb.shape[1])])
        embeddings.insert(0,'event_id',[g.event_id for g in ge])
        embeddings.to_csv(savedir/'event_embeddings.csv',index=False)
        # Actual WALLET-node embeddings from message passing, keyed by source/chain/address.
        wallet_rows=[];offset=0
        for g in ge:
            for j,node in enumerate(g.nodes):
                if node['type']=='wallet':
                    wallet_rows.append({'wallet_node_id':node['id'],'source':g.source,
                                        'snapshot_timestamp':str(g.timestamp),
                       **{f'embedding_{k}':float(v) for k,v in enumerate(node_embeddings[offset+j])}})
            offset += len(g.nodes)
        if wallet_rows:
            pd.DataFrame(wallet_rows).sort_values('snapshot_timestamp').drop_duplicates('wallet_node_id',keep='last').to_csv(savedir/'wallet_embeddings.csv',index=False)
    return model,th,scores


def run_training(processed='data/processed',source='uniswap',outdir='reports',modeldir='models',epochs=15,strict=True,max_events=1200):
    random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.set_num_threads(2)
    trades,transfers=load_processed(processed)
    if source not in trades.source.unique():
        raise ValueError(f"Source {source!r} not imported. Available: {list(trades.source.unique())}")
    trades=trades[trades.source==source].sort_values('timestamp').head(max_events).reset_index(drop=True)
    if len(transfers):
        transfers=transfers[transfers.chain.isin(trades.chain.unique())].copy()
    f=prepare_features(trades,transfers)
    splits=chrono_split(f)
    out=Path(outdir)/source;out.mkdir(parents=True,exist_ok=True)
    mdir=Path(modeldir)/source;mdir.mkdir(parents=True,exist_ok=True)
    f.to_csv(out/'features_and_weak_labels.csv',index=False)
    columns=STRICT if strict else NUMERIC
    train=splits['train'];val=splits['validation'];test=splits['test']
    fit=train[train.weak_label>=0];valid=val[val.weak_label>=0];held=test[test.weak_label>=0]
    models={
        'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',random_state=7,max_iter=350)),
        'random_forest':RandomForestClassifier(n_estimators=90,max_depth=7,class_weight='balanced',random_state=7,n_jobs=2),
    }
    metrics={};results=[]
    for name,m in models.items():
        m.fit(fit[columns].fillna(0),fit.weak_label)
        val_s=m.predict_proba(valid[columns].fillna(0))[:,1]
        threshold=_threshold(valid.weak_label.to_numpy(),val_s)
        score=m.predict_proba(held[columns].fillna(0))[:,1]
        metrics[name]=metric_report(held.weak_label,score,threshold)
        joblib.dump({'model':m,'features':columns,'threshold':threshold},mdir/(name+'.joblib'))
        results.append(pd.DataFrame({'event_id':held.event_id, 'risk_score':score,'model':name,
                          'weak_label':held.weak_label,'decision':np.where(score>=threshold,'suspicious_candidate','organic_candidate')}))
    graphs=build_graphs(trades,f,max_past=20,max_events=max_events)
    gmap={g.event_id:g for g in graphs}
    for name in ('gcn','sage'):
        model,threshold,score_data=train_graph_model(name,gmap,splits,mdir,epochs)
        metrics[name]=score_data['test']
        metrics[name]['training_seconds']=score_data['training_seconds']
        held_ids=[e for e in held.event_id]
        held_graphs=[gmap[e] for e in held_ids]
        x,ei,roots,y=batch_graphs(held_graphs)
        with torch.no_grad():
            scores=torch.sigmoid(model(x,ei,roots)[0]).numpy()
        results.append(pd.DataFrame({'event_id':held_ids,'risk_score':scores,'model':name,
                          'weak_label':held.weak_label.values,'decision':np.where(scores>=threshold,'suspicious_candidate','organic_candidate')}))
    predictions=pd.concat(results,ignore_index=True)
    predictions.to_csv(out/'test_predictions.csv',index=False)
    # Full overview after training: risk scores for held-out only are NOT presented as full coverage.
    groups=analyze_groups(trades,f,transfers)
    groups.to_csv(out/'candidate_wallet_groups.csv',index=False)
    (out/'metrics.json').write_text(json.dumps({'dataset':source,'label_definition':'heuristic_v1_past_only; weak labels NOT verified fraud',
         'strict_leakage_sensitivity_mode':strict,'feature_columns':columns,
         'split_counts':{k:len(v) for k,v in splits.items()},
         'cutoffs':{k:[str(v.timestamp.min()),str(v.timestamp.max())] for k,v in splits.items()},
         'models':metrics},indent=2))
    _charts(metrics,predictions,out)
    print('Saved computed WEAK-LABEL agreement measurements (NOT confirmed fraud) on',source,'to',out)
    for model,r in metrics.items():print(f"{model}: F1={r['f1']:.3f}, PR-AUC={r['pr_auc']}, positives={r['weak_positive_count']}")
    return metrics


def _charts(metrics,predictions,out):
    names=list(metrics)
    fig,ax=plt.subplots(figsize=(8,4))
    xx=np.arange(len(names))
    for i,k in enumerate(('f1','macro_f1','pr_auc')):
        vals=[metrics[m][k] if metrics[m][k] is not None else 0 for m in names]
        ax.bar(xx+(i-1)*.24,vals,.23,label=k)
    ax.set_xticks(xx,names);ax.set_ylim(0,1.05);ax.set_ylabel('Agreement with weak candidate labels')
    ax.legend();fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=145);plt.close(fig)
    best='sage'  # Predeclared model; never select the best model using test labels
    cm=np.asarray(metrics[best]['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(4,4));im=ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set_xticks([0,1],['Organic candidate','Suspicious candidate'],rotation=22)
    ax.set_yticks([0,1],['Organic candidate','Suspicious candidate']);ax.set_xlabel('Predicted');ax.set_ylabel('Weak candidate label')
    ax.set_title(best+' - test set');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=145);plt.close(fig)
    # PR curve on heldout for best model, not ROC on unverified real fraud labels.
    data=predictions[predictions.model==best]
    p,r,_=precision_recall_curve(data.weak_label,data.risk_score)
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(r,p);ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_title('PR curve - weak labels ('+best+')')
    ax.set_ylim(0,1.05);fig.tight_layout();fig.savefig(out/'precision_recall_curve.png',dpi=145);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed',default='data/processed');p.add_argument('--source',default='uniswap',choices=['idex','etherdelta','uniswap','pancakeswap'])
    p.add_argument('--outdir',default='reports');p.add_argument('--modeldir',default='models')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--max-events',type=int,default=1200)
    p.add_argument('--expanded-features',action='store_true',help='Including rule-correlated features; report leakage caveat')
    a=p.parse_args();run_training(a.processed,a.source,a.outdir,a.modeldir,a.epochs,not a.expanded_features,a.max_events)

if __name__=='__main__':main()
