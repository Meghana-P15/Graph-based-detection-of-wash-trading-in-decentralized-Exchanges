"""Offline-ready ASTRAQUANT PS18 investigation dashboard, load only real saved predictions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st


# HTML tables: avoid Streamlit's dataframe/table and st.write(dict) PyArrow conversion.
def render_prediction_table(table_df: pd.DataFrame, table_key: str, rows_per_page: int = 25):
    import math
    if table_df.empty:
        st.info("No rows to display.")
        return
    total_pages = max(1, math.ceil(len(table_df) / rows_per_page))
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1,
        step=1, key=f"{table_key}_page"
    )
    start = (page - 1) * rows_per_page
    subset = table_df.iloc[start : start + rows_per_page].copy()
    # Escape user-supplied dataset text, so HTML content cannot execute.
    table_html = subset.to_html(index=False, escape=True, border=0, classes="astra-table")
    st.html(
        """
        <style>
        .astra-scroll {overflow-x:auto; max-height:480px; overflow-y:auto;
            border:1px solid rgba(140,140,140,.35); border-radius:9px;}
        .astra-table {border-collapse:collapse; width:100%; font-size:0.84rem;}
        .astra-table th,.astra-table td {padding:9px 12px; border-bottom:1px solid rgba(140,140,140,.25);
            text-align:left; white-space:nowrap;}
        .astra-table thead th {position:sticky; top:0; background:#26303b; color:#fff; z-index:1;}
        .astra-table tbody tr:nth-child(even) {background:rgba(128,128,128,.07);}
        </style>
        """ + '<div class="astra-scroll">' + table_html + '</div>',
    )
    st.caption(f"Showing {start + 1}–{min(start + rows_per_page, len(table_df))} of {len(table_df)} rows")

st.set_page_config(page_title='ASTRAQUANT PS18 - DEX Investigation',page_icon='🔎',layout='wide')
st.title('ASTRAQUANT X 2026 | DEX Wash Trading Investigation')
st.caption('Research prototype: suspicious vs organic *candidates*. A model score does not establish fraud, ownership, or intent.')

root=Path(__file__).resolve().parents[1]
opts=[p.parent.name for p in (root/'reports').glob('*/inference_predictions.csv')]
if not opts:
    st.warning('No saved predictions. Run `python -m src.train --source uniswap` followed by `python -m src.infer --source uniswap --arch sage`.')
    st.stop()
source=st.sidebar.selectbox('DEX dataset',sorted(opts))
base=root/'reports'/source
f=base/'inference_predictions.csv'
pred=pd.read_csv(f)
trades=pd.read_csv(root/'data/processed/trades.csv')
trades=trades[trades.source==source]
df=pred.merge(trades[['event_id','tx_hash','chain','protocol','pool','counterparty_wallet',
                      'token_in','token_out','amount_in','amount_out','amount_usd']],on='event_id',how='left')
synthetic=(root/'data/processed/data_audit.json').exists() and 'data/synthetic/' in (root/'data/processed/data_audit.json').read_text()
if synthetic:
    st.error('SYNTHETIC DEMO DATA: model outputs are generated from synthetic fixtures, not real blockchain evidence.')
else:
    st.info('Unverified research-derived weak labels. Flagged activity requires investigator confirmation.')

c1,c2,c3,c4=st.columns(4)
c1.metric('Analyzed events',len(df))
c2.metric('Suspicious candidates',int((df.candidate_decision=='suspicious_candidate').sum()))
c3.metric('Review required',int(df.review_required.sum()))
c4.metric('Organic candidates',int((df.candidate_decision=='organic_candidate').sum()))

st.subheader('Transaction investigation')
choices=st.selectbox('Select transaction ID',df.event_id.tolist(),format_func=lambda x:x[-34:])
r=df.loc[df.event_id==choices].iloc[0]
left,right=st.columns([1,1])
with left:
    st.metric('Investigation risk score',f"{r.risk_score:.3f}")
    st.write('**Candidate classification:**',r.candidate_decision)
    st.write('**DEX:**',r.protocol,' | **Chain:**',r.chain)
    st.write('**Actor (observed):**',r.actor_wallet)
    st.write('**Tx hash:**',r.tx_hash)
    st.write('**Time (UTC):**',r.timestamp)
    st.write('**Amount USD (if present):**',r.amount_usd)
    st.markdown('**Current weak-label evidence:**')
    st.code(json.dumps(json.loads(r.label_reason_codes), indent=2), language='json')
    st.markdown('**Transfer paths (observed before trade):**')
    st.code(json.dumps(json.loads(r.transfer_paths), indent=2), language='json')
    st.warning('Do not infer that associated addresses share a real owner. Routing and arbitrage are plausible benign alternatives.')
with right:
    st.markdown('**Observed evidence graph (selected transaction context)**')
    G=nx.Graph()
    actor=str(r.actor_wallet);G.add_node(actor,kind='wallet')
    event=str(r.event_id);G.add_node(event,kind='trade');G.add_edge(actor,event,relation='actor-performed-trade')
    if pd.notna(r.counterparty_wallet) and str(r.counterparty_wallet):
        c=str(r.counterparty_wallet);G.add_node(c,kind='wallet');G.add_edge(c,event,relation='observed-maker')
    if pd.notna(r.pool) and str(r.pool):
        p=str(r.pool);G.add_node(p,kind='pool');G.add_edge(event,p,relation='swap-uses-pool')
    for token in [r.token_in,r.token_out]:
        if pd.notna(token) and str(token):
            G.add_node(str(token),kind='token');G.add_edge(event,str(token),relation='trade-token')
    paths=json.loads(r.transfer_paths)
    for path in paths:
        for a,b in zip(path.get('wallets',[]),path.get('wallets',[])[1:]):
            G.add_node(a,kind='wallet');G.add_node(b,kind='wallet');G.add_edge(a,b,relation='observed-token-transfer')
    pos=nx.spring_layout(G,seed=7)
    ex,ey=[],[]
    for a,b in G.edges():
        ex.extend([pos[a][0],pos[b][0],None]);ey.extend([pos[a][1],pos[b][1],None])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ex,y=ey,mode='lines',line=dict(color='#8b9baa',width=2),hoverinfo='none'))
    colors={'wallet':'#07539B','trade':'#008C8C','pool':'#a78bfa','token':'#f59e0b'}
    for kind in colors:
        nodes=[n for n in G if G.nodes[n].get('kind')==kind]
        if nodes:
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],
             text=[n if len(n)<32 else n[:9]+'...'+n[-7:] for n in nodes],hovertext=nodes,
             name=kind,mode='markers+text',textposition='bottom center',marker=dict(size=15,color=colors[kind])))
    fig.update_layout(height=400,margin=dict(l=5,r=5,t=5,b=5),showlegend=True,xaxis=dict(visible=False),yaxis=dict(visible=False))
    st.plotly_chart(fig,width="stretch")

st.subheader('Transaction predictions')
render_prediction_table(
    df[['event_id', 'timestamp', 'candidate_decision', 'risk_score', 'review_required']]
    .sort_values('risk_score', ascending=False),
    table_key='predictions',
)
st.download_button('Export predictions CSV',df.to_csv(index=False).encode(),file_name=source+'_investigation_predictions.csv',mime='text/csv')
metrics=base/'metrics.json'
if metrics.exists():
    st.subheader('Weak-label comparison (NOT verified fraud accuracy)')
    st.json(json.loads(metrics.read_text())['models'])
group=base/'candidate_wallet_groups.csv'
if group.exists():
    st.subheader('Candidate coordinated groups')
    render_prediction_table(pd.read_csv(group), table_key='groups')
st.divider()
st.caption('Safety: public/pseudonymous wallet data; no automated blocking; no common-owner conclusions; human review required.')
