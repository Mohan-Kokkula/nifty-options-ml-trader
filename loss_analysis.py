import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '.')
from core.ml_engine import load_model
from backtest import _load_csv, _build_full_features, SL_POINTS, TARGET_POINTS, LOT_SIZE, DELTA_APPROX, DAILY_LOSS_CAP, SIGNAL_FLOOD_LIMIT

load_model('.')
df_full = _load_csv('data/nifty_5min.csv')
signals, probas, feat = _build_full_features(df_full)

SL=SL_POINTS; TP=TARGET_POINTS; LOT=LOT_SIZE; D=DELTA_APPROX
trades=[]
pos_dir=None; entry_price=0; sl_level=0; tp_level=0
current_day=None; daily_pnl=0.0; daily_halted=False
pre1300_signals=0; flood_halted=False
entry_conf=0.0; entry_hour=0; entry_vix=0.0; entry_adx=0.0; entry_dow=0

_vix_arr=feat['vix_level'].values if 'vix_level' in feat.columns else np.zeros(len(feat))
_adx_arr=feat['tf15_adx'].values if 'tf15_adx' in feat.columns else np.zeros(len(feat))
_close_arr=feat['close'].values
_conf_arr=np.array([max(probas[i][0], probas[i][1]) for i in range(len(probas))])

for i in range(len(feat)):
    ts=feat.index[i]; spot=float(_close_arr[i]); sig=signals[i]
    conf=float(_conf_arr[i])
    vix=float(_vix_arr[i])
    adx=float(_adx_arr[i])

    if ts.date()!=current_day:
        current_day=ts.date(); daily_pnl=0.0; daily_halted=False
        pre1300_signals=0; flood_halted=False

    if daily_halted:
        if pos_dir:
            move=spot-entry_price
            pnl=(D*(move if pos_dir=='CALL' else -move))*LOT
            trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':pos_dir,'reason':'DAILY_CAP','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
            daily_pnl+=pnl; pos_dir=None
        continue

    if flood_halted:
        if pos_dir:
            move=spot-entry_price
            pnl=(D*(move if pos_dir=='CALL' else -move))*LOT
            trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':pos_dir,'reason':'FLOOD_EXIT','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
            daily_pnl+=pnl; pos_dir=None
        continue

    if ts.hour>=15 and ts.minute>=20:
        if pos_dir:
            move=spot-entry_price
            pnl=(D*(move if pos_dir=='CALL' else -move))*LOT
            trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':pos_dir,'reason':'SQUARE_OFF','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
            daily_pnl+=pnl; pos_dir=None
        continue

    if pos_dir:
        if pos_dir=='CALL':
            if spot<=sl_level:
                pnl=(D*(spot-entry_price))*LOT
                trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':'CALL','reason':'SL','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
                daily_pnl+=pnl; pos_dir=None
                if daily_pnl<=-DAILY_LOSS_CAP: daily_halted=True
            elif spot>=tp_level:
                pnl=(D*(spot-entry_price))*LOT
                trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':'CALL','reason':'TP','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
                daily_pnl+=pnl; pos_dir=None
        else:
            if spot>=sl_level:
                pnl=(D*(entry_price-spot))*LOT
                trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':'PUT','reason':'SL','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
                daily_pnl+=pnl; pos_dir=None
                if daily_pnl<=-DAILY_LOSS_CAP: daily_halted=True
            elif spot<=tp_level:
                pnl=(D*(entry_price-spot))*LOT
                trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':'PUT','reason':'TP','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
                daily_pnl+=pnl; pos_dir=None

    if sig==2: continue

    if ts.hour<13:
        pre1300_signals+=1
        if pre1300_signals>SIGNAL_FLOOD_LIMIT:
            flood_halted=True
            if pos_dir:
                move=spot-entry_price
                pnl=(D*(move if pos_dir=='CALL' else -move))*LOT
                trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':pos_dir,'reason':'FLOOD_EXIT','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
                daily_pnl+=pnl; pos_dir=None
            continue

    sd='CALL' if sig==0 else 'PUT'
    if pos_dir is None:
        pos_dir=sd; entry_price=spot; entry_conf=conf; entry_hour=ts.hour; entry_vix=vix; entry_adx=adx; entry_dow=ts.dayofweek
        sl_level=spot-SL if sd=='CALL' else spot+SL
        tp_level=spot+TP if sd=='CALL' else spot-TP
    elif pos_dir!=sd:
        move=spot-entry_price
        pnl=(D*(move if pos_dir=='CALL' else -move))*LOT
        trades.append({'time':ts,'year':ts.year,'pnl':pnl,'dir':pos_dir,'reason':'REVERSAL','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})
        daily_pnl+=pnl; pos_dir=sd; entry_price=spot; entry_conf=conf; entry_hour=ts.hour; entry_vix=vix; entry_adx=adx; entry_dow=ts.dayofweek
        sl_level=spot-SL if sd=='CALL' else spot+SL
        tp_level=spot+TP if sd=='CALL' else spot-TP

if pos_dir:
    spot=float(feat['close'].iloc[-1]); move=spot-entry_price
    pnl=(D*(move if pos_dir=='CALL' else -move))*LOT
    trades.append({'time':feat.index[-1],'year':feat.index[-1].year,'pnl':pnl,'dir':pos_dir,'reason':'END','conf':entry_conf,'hour':entry_hour,'vix':entry_vix,'adx':entry_adx,'dow':entry_dow})

tdf=pd.DataFrame(trades)
losses=tdf[tdf.pnl<=0]
wins=tdf[tdf.pnl>0]

print(f'Total trades: {len(tdf)} | Winners: {len(wins)} ({len(wins)/len(tdf)*100:.1f}%) | Losers: {len(losses)} ({len(losses)/len(tdf)*100:.1f}%)')
print()

print('=== LOSSES BY EXIT REASON ===')
r=tdf.groupby('reason').apply(lambda g: pd.Series({'trades':len(g),'wins':int((g.pnl>0).sum()),'losses':int((g.pnl<=0).sum()),'win_pct':round((g.pnl>0).mean()*100,1),'total_pnl':int(g.pnl.sum())}))
print(r.to_string())
print()

print('=== LOSSES BY ENTRY HOUR ===')
bins=[9,10,11,12,13,14,15,16]
tdf['hour_bin']=pd.cut(tdf['hour'],bins=bins,labels=['9-10','10-11','11-12','12-13','13-14','14-15','15+'],right=False)
h=tdf.groupby('hour_bin',observed=True).apply(lambda g: pd.Series({'trades':len(g),'win_pct':round((g.pnl>0).mean()*100,1),'total_pnl':int(g.pnl.sum())}))
print(h.to_string())
print()

print('=== LOSSES BY VIX LEVEL ===')
vbins=[0,12,16,20,22,25,30,100]
vlabels=['<12','12-16','16-20','20-22','22-25','25-30','>30']
tdf['vix_bin']=pd.cut(tdf['vix'],bins=vbins,labels=vlabels)
v=tdf.groupby('vix_bin',observed=True).apply(lambda g: pd.Series({'trades':len(g),'win_pct':round((g.pnl>0).mean()*100,1),'total_pnl':int(g.pnl.sum())}))
print(v.to_string())
print()

print('=== LOSSES BY ML CONFIDENCE ===')
cbins=[0,0.30,0.35,0.40,0.45,0.50,0.60,1.0]
clabels=['<30','30-35','35-40','40-45','45-50','50-60','>60']
tdf['conf_bin']=pd.cut(tdf['conf'],bins=cbins,labels=clabels)
c=tdf.groupby('conf_bin',observed=True).apply(lambda g: pd.Series({'trades':len(g),'win_pct':round((g.pnl>0).mean()*100,1),'total_pnl':int(g.pnl.sum())}))
print(c.to_string())
print()

print('=== LOSSES BY ADX ===')
abins=[0,15,20,25,30,40,60,100]
alabels=['<15','15-20','20-25','25-30','30-40','40-60','>60']
tdf['adx_bin']=pd.cut(tdf['adx'],bins=abins,labels=alabels)
a=tdf.groupby('adx_bin',observed=True).apply(lambda g: pd.Series({'trades':len(g),'win_pct':round((g.pnl>0).mean()*100,1),'total_pnl':int(g.pnl.sum())}))
print(a.to_string())
print()

print('=== LOSSES BY DAY OF WEEK ===')
days={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}
tdf['day']=tdf['dow'].map(days)
d=tdf.groupby('day',observed=True).apply(lambda g: pd.Series({'trades':len(g),'win_pct':round((g.pnl>0).mean()*100,1),'total_pnl':int(g.pnl.sum())}))
print(d.to_string())
