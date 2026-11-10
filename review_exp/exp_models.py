import sys, warnings, pandas as pd, numpy as np
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, recall_score, precision_score, cohen_kappa_score, accuracy_score, r2_score
SEED=42
CAT=["shift","line_id","product_family","order_priority"]; NUM=["planned_qty","available_operators","planned_lead_time_min"]
BASE=CAT+NUM; TOP5=["planned_lead_time_min","available_operators","line_id","product_family","shift"]
EVENT=["changeover_min","material_wait_min","equipment_stop_min"]
d=pd.read_csv(sys.argv[1],encoding='utf-8-sig',parse_dates=['lot_start_time'])
cut=lambda s: np.select([s<18,s<35,s<60],[0,1,2],3)
d['y']=cut(d.schedule_delay_min)
o=d.sort_values(['lot_start_time','record_id']).reset_index(drop=True)
def split(df,f):
    c=df.iloc[int(len(df)*f)].lot_start_time
    old=df[df.lot_start_time<c]; fut=df[df.lot_start_time>=c]
    done=old.lot_start_time+pd.to_timedelta(old.actual_lead_time_min,unit='m')
    return old[done<=c], fut
tr,va=split(o,0.75)
print('split',len(tr),len(va))
def pre(cols):
    cat=[c for c in cols if c in CAT]; num=[c for c in cols if c not in CAT]
    b=[]
    if cat: b.append(('c',Pipeline([('f',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    if num: b.append(('n',Pipeline([('f',SimpleImputer(strategy='median')),('s',StandardScaler())]),num))
    return ColumnTransformer(b,sparse_threshold=0)
def clf(alg):
    return {'LR':lambda:LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED),
            'RF':lambda:RandomForestClassifier(n_estimators=90,min_samples_leaf=3,class_weight='balanced_subsample',max_features='sqrt',n_jobs=-1,random_state=SEED)}[alg]()
def met(y,p):
    return dict(F1=f1_score(y,p,average='macro'),FR=recall_score(y==3,p==3),FP=precision_score(y==3,p==3,zero_division=0),QWK=cohen_kappa_score(y,p,weights='quadratic'),acc=accuracy_score(y,p),nF=int((p==3).sum()))
rows=[]
def run(name,cols,alg,a=tr,b=va):
    m=Pipeline([('p',pre(cols)),('m',clf(alg))]).fit(a[cols],a.y); p=m.predict(b[cols]); rows.append((name,met(b.y.values,p))); return m
run('RF 7개',BASE,'RF'); run('RF Top5',TOP5,'RF'); run('LR 7개',BASE,'LR'); m_lr5=run('LR Top5',TOP5,'LR')
# regression then cut
for nm,reg,cols in [('Ridge회귀→컷',Ridge(alpha=1.0),BASE),('Ridge회귀 Top5→컷',Ridge(alpha=1.0),TOP5),('RF회귀→컷',RandomForestRegressor(300,min_samples_leaf=10,random_state=SEED,n_jobs=-1),BASE),('HGB회귀→컷',HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_leaf_nodes=9,random_state=SEED),BASE)]:
    m=Pipeline([('p',pre(cols)),('m',reg)]).fit(tr[cols],tr.schedule_delay_min)
    ph=m.predict(va[cols]); print(nm,'R2 valid=%.3f'%r2_score(va.schedule_delay_min,ph),'train R2=%.3f'%r2_score(tr.schedule_delay_min,m.predict(tr[cols])))
    rows.append((nm+'(고정컷)',met(va.y.values,cut(ph))))
    # tune cutoffs on train predictions via quantile matching to train class prior
    pt=m.predict(tr[cols]); q=np.cumsum(np.bincount(tr.y,minlength=4))/len(tr)
    cuts=np.quantile(pt,q[:3]); pv=np.searchsorted(cuts,ph,side='right')
    rows.append((nm+'(분포맞춤컷)',met(va.y.values,pv)))
# threshold on failure prob for LR Top5 and RF Top5
for nm,cols,alg in [('LR Top5',TOP5,'LR'),('RF Top5',TOP5,'RF')]:
    m=Pipeline([('p',pre(cols)),('m',clf(alg))]).fit(tr[cols],tr.y); P=m.predict_proba(va[cols])
    for t in [0.3,0.4,0.5]:
        p=P.argmax(1).copy(); p[P[:,3]>=t]=3; p[(P.argmax(1)==3)&(P[:,3]<t)]=np.argsort(P,1)[:,-2][(P.argmax(1)==3)&(P[:,3]<t)]
        rows.append((f'{nm} Failure컷 {t}',met(va.y.values,p)))
# headroom: with event variables (post-hoc / real-time info)
run('RF 7개+실측3개(상한 참고)',BASE+EVENT,'RF'); run('LR 7개+실측3개(상한 참고)',BASE+EVENT,'LR')
r=pd.DataFrame({k:v for k,v in rows}).T
print(r.round(3).to_string())
# expanding 3-fold for LR Top5 vs RF Top5 vs LR7 vs RF7
n=len(o); res={}
for k,(fr,to) in enumerate([(0.4,0.6),(0.6,0.8),(0.8,1.0)]):
    c1=o.iloc[int(n*fr)].lot_start_time; c2=o.iloc[min(int(n*to),n-1)].lot_start_time if to<1 else o.lot_start_time.max()+pd.Timedelta('1s')
    old=o[o.lot_start_time<c1]; done=old.lot_start_time+pd.to_timedelta(old.actual_lead_time_min,unit='m'); a=old[done<=c1]
    b=o[(o.lot_start_time>=c1)&(o.lot_start_time<c2)]
    for nm,cols,alg in [('LR 7개',BASE,'LR'),('RF 7개',BASE,'RF'),('LR Top5',TOP5,'LR'),('RF Top5',TOP5,'RF')]:
        m=Pipeline([('p',pre(cols)),('m',clf(alg))]).fit(a[cols],a.y); res.setdefault(nm,[]).append(f1_score(b.y,m.predict(b[cols]),average='macro'))
print({k:[round(x,3) for x in v]+[round(np.mean(v),4)] for k,v in res.items()})
