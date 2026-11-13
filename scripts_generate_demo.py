"""Generate clearly SYNTHETIC tiny schema fixtures for smoke testing, not real DEX trades."""
import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

TOKA='0x'+'1'*40; TOKB='0x'+'2'*40
POOLE='0x'+'3'*40; POOLB='0x'+'4'*40
ADDRESS=lambda i:'0x'+format(i,'040x')


def create(output="data/synthetic", days=4, seed=19):
    rng=random.Random(seed); out=Path(output);out.mkdir(parents=True,exist_ok=True)
    decimals={TOKA:{"address":TOKA,"decimals":18},TOKB:{"address":TOKB,"decimals":18}}
    (out/"token_decimals.json").write_text(json.dumps(decimals,indent=2))
    for source in ("idex","etherdelta","uniswap","pancakeswap"):
        historical=source in ("idex","etherdelta")
        base=datetime(2019,1,1,tzinfo=timezone.utc) if historical else datetime(2026,10,1,tzinfo=timezone.utc)
        records=[];transfer_rows=[];counter=0
        for hour in range(days*24):
            # Independent organic activity, widely separated wallets and token trading.
            for j in range(2):
                tm=base+timedelta(hours=hour,minutes=(j*17+3))
                actor=ADDRESS(1000+hour*5+j)
                maker=ADDRESS(2000+hour*5+j)
                in_t,out_t=(TOKA,TOKB) if j%2==0 else (TOKB,TOKA)
                records.append(_row(source,counter,tm,actor,maker,in_t,out_t,100.+j*17,rng));counter+=1
            # Suspicious TEST SCENARIO: burst of repeated and reversing trades.
            if hour%3 != 1:
                group=[ADDRESS(10+k) for k in range(3)]
                for j in range(7):
                    tm=base+timedelta(hours=hour, minutes=35+j*2)
                    actor=group[(hour+j)%3]
                    maker=group[(hour+j+1)%3]
                    in_t,out_t=(TOKA,TOKB) if j%2==0 else (TOKB,TOKA)
                    records.append(_row(source,counter,tm,actor,maker,in_t,out_t,150.+(j%2)*2,rng));counter+=1
                if not historical:
                    # A -> B -> C, strictly ordered transfers precede later swaps.
                    chain='ethereum' if source=='uniswap' else 'bnb'
                    for k,(a,b) in enumerate(zip(group[:2],group[1:])):
                        transfer_rows.append(dict(blockchain=chain,block_time=(base+timedelta(hours=hour,minutes=21+k*2)).isoformat(),
                             tx_hash='0x'+format(900000+hour*10+k+(0 if source=='uniswap' else 100000),'064x'),evt_index=k,
                             from_address=a,to_address=b,token_address=TOKA,symbol='TOKA',amount=10.,amount_usd=10.))
        pd.DataFrame(records).to_csv(out/(source+'.csv'),index=False)
        if transfer_rows:
            name='ethereum_transfers.csv' if source=='uniswap' else 'bnb_transfers.csv'
            pd.DataFrame(transfer_rows).to_csv(out/name,index=False)
    (out/"README_SYNTHETIC.txt").write_text("ALL FILES HERE ARE GENERATED SYNTHETIC FIXTURES. NOT ACTUAL BLOCKCHAIN EVENTS OR REAL FRAUD LABELS. DO NOT USE FOR REVIEW 3 REAL-DATA ACCURACY CLAIMS.\n")
    return out


def _row(source,n,tm,actor,maker,token_in,token_out,amount,rng):
    historical=source in ('idex','etherdelta')
    tx='0x'+format(n+1,'064x')
    if source=='idex':
        # maker-order tokenBuy=out, tokenSell=in; actual filled tokenBuy 1-2 units.
        amt=str(int((1.0+rng.random())*10**18))
        return dict(transaction_hash=tx,status=1,block_number=1000+n,timestamp=int(tm.timestamp()),
            maker=maker,taker=actor,tokenBuy=token_out,tokenSell=token_in,
            amountBuy=str(2*10**18),amountSell=str(int(amount*2*10**18)),amount=amt)
    if source=='etherdelta':
        return dict(transaction_hash=tx,block_number=1000+n,timestamp=int(tm.timestamp()),
            get=maker,give=actor,tokenGet=token_out,tokenGive=token_in,
            amountGet=str(10**18), amountGive=str(int(amount*10**18)))
    return dict(blockchain='ethereum' if source=='uniswap' else 'bnb',
        project='uniswap' if source=='uniswap' else 'pancakeswap',version=3,
        block_time=tm.isoformat(),block_number=60000000+n,tx_hash=tx,evt_index=0,
        tx_from=actor,tx_to=POOLE if source=='uniswap' else POOLB,
        taker=actor,maker=None,
        project_contract_address=POOLE if source=='uniswap' else POOLB,
        token_bought_address=token_out,token_sold_address=token_in,
        token_bought_symbol='TOK',token_sold_symbol='TOK',token_bought_amount=1.,
        token_sold_amount=1., amount_usd=amount)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/synthetic');p.add_argument('--days',type=int,default=4)
    a=p.parse_args();print('Synthetic fixtures created at',create(a.output,a.days))
