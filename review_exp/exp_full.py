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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import lightgbm as lgb
def clf2(alg):
    if alg=='HGB': return HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=1,class_weight='balanced',random_state=SEED)
    if alg=='LGBM': return lgb.LGBMClassifier(n_estimators=300,num_leaves=15,learning_rate=0.05,class_weight='balanced',random_state=SEED,verbosity=-1,n_jobs=1)
    return clf(alg)
def run(name,cols,alg,a=None,b=None):
    a=tr if a is None else a; b=va if b is None else b
    m=Pipeline([('p',pre(cols)),('m',clf2(alg))]).fit(a[cols],a.y); p=m.predict(b[cols]); rows.append((name,met(b.y.values,p))); return m
# leakage: actual_lead_time_min(=planned+delay), schedule_delay, on_time flag. bottleneck_station = outcome label (exclude). actual_qty? check
print('actual_qty==planned_qty share', (d.actual_qty==d.planned_qty).mean())
print('delay==actual-planned', np.allclose(d.actual_lead_time_min-d.planned_lead_time_min, d.schedule_delay_min, atol=0.15))
SENS=["changeover_min","station_a_cycle_sec","station_b_cycle_sec","station_c_cycle_sec","station_d_cycle_sec","wait_before_b_min","wait_before_c_min","wait_before_d_min","wip_start_qty","wip_end_qty","equipment_stop_min","material_wait_min","actual_qty"]
ALL=BASE+SENS
for alg in ['LR','RF','HGB','LGBM']: run(f'1단계 전체 {len(ALL)}개 {alg}',ALL,alg)
# importance
m=Pipeline([('p',pre(ALL)),('m',clf2('RF'))]).fit(tr[ALL],tr.y)
pi=permutation_importance(m,va[ALL],va.y,scoring='f1_macro',n_repeats=10,random_state=SEED)
imp=pd.Series(pi.importances_mean,ALL).sort_values(ascending=False); print(imp.round(4).to_string())
for k in [5,8,10]:
    top=imp.index[:k].tolist()
    for alg in ['LR','RF']: run(f'2단계 Top{k} {alg}',top,alg)
def enrich(x):
    x=x.copy(); q=x.planned_qty
    for s in 'abcd': x[f'{s}_proc_min']=x[f'station_{s}_cycle_sec']*q/60
    x['max_cycle']=x[[f'station_{s}_cycle_sec' for s in 'abcd']].max(1)
    x['bneck_proc_min']=x.max_cycle*q/60
    x['total_wait']=x[['wait_before_b_min','wait_before_c_min','wait_before_d_min']].sum(1,min_count=1)
    x['event_total']=x.changeover_min+x.equipment_stop_min+x.material_wait_min
    x['wip_delta']=x.wip_end_qty-x.wip_start_qty
    x['est_extra']=x.total_wait+x.event_total
    return x
tre,vae=enrich(tr),enrich(va)
DER=['bneck_proc_min','max_cycle','total_wait','event_total','wip_delta','est_extra']
for alg in ['LR','RF','HGB']: run(f'3단계 전체+파생 {alg}',ALL+DER,alg,tre,vae)
for alg in ['LR','RF']: run(f'3단계 Top8+파생 {alg}',imp.index[:8].tolist()+DER,alg,tre,vae)
# regression check
from sklearn.linear_model import Ridge
for nm,cols,a,b in [('전체',ALL,tr,va),('전체+파생',ALL+DER,tre,vae)]:
    r=Pipeline([('p',pre(cols)),('m',Ridge())]).fit(a[cols],a.schedule_delay_min); print(nm,'Ridge R2 %.3f'%r2_score(b.schedule_delay_min,r.predict(b[cols])))
r=pd.DataFrame({k:v for k,v in rows}).T; print(r.round(3).to_string())
