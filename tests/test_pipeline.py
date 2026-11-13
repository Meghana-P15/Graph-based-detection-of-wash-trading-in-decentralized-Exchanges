import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts_generate_demo import create
from src.data import normalize,load_decimals,normalize_transfers
from src.features import prepare_features,find_prior_transfer_paths
from src.graph import make_graph_for_event,build_graphs,batch_graphs
from src.model import EventGNN

@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    p=tmp_path_factory.mktemp('samples');create(p,days=2)
    return p


def test_all_source_adapters_real_style_schemas(fixtures):
    dec=load_decimals(fixtures/'token_decimals.json')
    for source in ('idex','etherdelta','uniswap','pancakeswap'):
        raw=pd.read_csv(fixtures/(source+'.csv'))
        normalized=normalize(raw,source,dec)
        assert len(normalized)==len(raw)
        assert normalized.event_id.is_unique
        assert normalized.timestamp.notna().all()
        assert normalized.actor_wallet.str.startswith('0x').all()
        assert normalized.amount_in.notna().all()
        if source in ('uniswap','pancakeswap'):
            assert normalized.counterparty_wallet.eq('').all(), 'AMM must NOT create maker/taker wallet trading edges'
            assert normalized.pool.ne('').all()
        else:
            assert normalized.counterparty_wallet.ne('').all()
            assert normalized.pool.eq('').all()


def test_raw_historical_without_decimals_not_claim_normalized(fixtures):
    raw=pd.read_csv(fixtures/'idex.csv').head(3)
    o=normalize(raw,'idex',{})
    assert o.amount_in.isna().all()
    assert (~o.amounts_normalized).all()


def test_past_only_features_unaffected_by_future_event(fixtures):
    raw=pd.read_csv(fixtures/'uniswap.csv')
    t=normalize(raw,'uniswap');f=prepare_features(t)
    extra=t.tail(1).copy();extra['timestamp']=t.timestamp.max()+pd.Timedelta(hours=48);extra['event_id']='extra_event'
    more=prepare_features(pd.concat([t,extra],ignore_index=True))
    cols=['prior_actor_1h','prior_pair_1h','weak_label','prior_pool_1h']
    pd.testing.assert_frame_equal(f[cols],more.head(len(t))[cols])


def test_transfer_paths_ordered_and_chain_scoped(fixtures):
    raw=pd.read_csv(fixtures/'ethereum_transfers.csv');trans=normalize_transfers(raw,'ethereum')
    c='0x'+format(12,'040x')
    cutoff=trans.timestamp.min()+pd.Timedelta(hours=1)
    paths=find_prior_transfer_paths(trans,c,'ethereum',cutoff)
    assert any(p['wallets'][-1]==c and p['hops']>=2 for p in paths)
    assert find_prior_transfer_paths(trans,c,'bnb',cutoff)==[]


def test_no_future_graph_edges_or_fictitious_amm_counterparty(fixtures):
    t=normalize(pd.read_csv(fixtures/'uniswap.csv').head(18),'uniswap')
    f=prepare_features(t);g=build_graphs(t,f)
    assert len(g)==len(t)
    assert not any(link['relation']=='event_observed_maker' for graph in g for link in graph.links)
    assert not any(link['relation']=='event_taker' for graph in g for link in graph.links)
    row=t.iloc[1]
    with pytest.raises(ValueError,match='Leakage'):
        make_graph_for_event(row,t.iloc[[2]],0)


def test_actual_gcn_and_graphsage_forward(fixtures):
    t=normalize(pd.read_csv(fixtures/'idex.csv').head(45),'idex',load_decimals(fixtures/'token_decimals.json'))
    f=prepare_features(t);batch=batch_graphs(build_graphs(t,f))
    for arch in ('gcn','sage'):
        m=EventGNN(arch)
        logits,emb=m(*batch[:3])
        assert logits.shape[0]==len(t)
        assert emb.shape[0]==len(t)
        assert np.isfinite(logits.detach().numpy()).all()


def test_wrong_chain_rejected(fixtures):
    raw=pd.read_csv(fixtures/'uniswap.csv').head(2)
    raw['blockchain']='bnb'
    with pytest.raises(ValueError,match='Unexpected blockchain'):
        normalize(raw,'uniswap')


def test_failed_idex_trade_excluded(fixtures):
    raw=pd.read_csv(fixtures/'idex.csv').head(3)
    raw.loc[raw.index[0],'status']=0
    out=normalize(raw,'idex',load_decimals(fixtures/'token_decimals.json'))
    assert len(out)==2
