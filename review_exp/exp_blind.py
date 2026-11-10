import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, recall_score, precision_score, cohen_kappa_score, accuracy_score
SEED=42
CAT=["shift","line_id","product_family","order_priority"]; NUM=["planned_qty","available_operators","planned_lead_time_min"]
BASE=CAT+NUM; TOP5=["planned_lead_time_min","available_operators","line_id","product_family","shift"]
SENS=["changeover_min","station_a_cycle_sec","station_b_cycle_sec","station_c_cycle_sec","station_d_cycle_sec","wait_before_b_min","wait_before_c_min","wait_before_d_min","wip_start_qty","wip_end_qty","equipment_stop_min","material_wait_min"]
ALL=BASE+SENS  # actual_qty 제외 (Blind 입력에 없음)
cut=lambda s: np.select([s<18,s<35,s<60],[0,1,2],3)
o=pd.read_csv(sys.argv[1],encoding='utf-8-sig',parse_dates=['lot_start_time']); o['y']=cut(o.schedule_delay_min)
bi=pd.read_csv(sys.argv[2],encoding='utf-8-sig'); ak=pd.read_csv(sys.argv[3],encoding='utf-8-sig')
b=bi.merge(ak,on='record_id',validate='one_to_one'); b['y']=cut(b.actual_schedule_delay_min)
print('blind',len(b),'grades',np.bincount(b.y).tolist(),'| actual_risk_level 일치:',pd.crosstab(b.y,b.actual_risk_level).to_dict() if 'actual_risk_level' in b else '')
o=o.sort_values(['lot_start_time','record_id']).reset_index(drop=True)
c=o.iloc[int(len(o)*.75)].lot_start_time; old=o[o.lot_start_time<c]
tr=old[(old.lot_start_time+pd.to_timedelta(old.actual_lead_time_min,unit='m'))<=c]; va=o[o.lot_start_time>=c]
def pre(cols):
    cat=[x for x in cols if x in CAT]; num=[x for x in cols if x not in CAT]; bl=[]
    if cat: bl.append(('c',Pipeline([('f',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    if num: bl.append(('n',Pipeline([('f',SimpleImputer(strategy='median')),('s',StandardScaler())]),num))
    return ColumnTransformer(bl,sparse_threshold=0)
def clf(a):
    return LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED) if a=='LR' else RandomForestClassifier(n_estimators=90,min_samples_leaf=3,class_weight='balanced_subsample',max_features='sqrt',n_jobs=-1,random_state=SEED)
def enrich(x):
    x=x.copy(); q=x.planned_qty
    x['max_cycle']=x[[f'station_{s}_cycle_sec' for s in 'abcd']].max(1)
    x['bneck_proc_min']=x.max_cycle*q/60
    x['total_wait']=x[['wait_before_b_min','wait_before_c_min','wait_before_d_min']].sum(1,min_count=1)
    x['event_total']=x.changeover_min+x.equipment_stop_min+x.material_wait_min
    x['wip_delta']=x.wip_end_qty-x.wip_start_qty
    x['est_extra']=x.total_wait+x.event_total
    return x
DER=['bneck_proc_min','max_cycle','total_wait','event_total','wip_delta','est_extra']
o,tr,va,b=map(enrich,[o,tr,va,b])
m=Pipeline([('p',pre(ALL)),('m',clf('RF'))]).fit(tr[ALL],tr.y)
imp=pd.Series(permutation_importance(m,va[ALL],va.y,scoring='f1_macro',n_repeats=10,random_state=SEED).importances_mean,ALL).sort_values(ascending=False)
TOP8=imp.index[:8].tolist(); print('Top8:',TOP8)
def met(y,p): return dict(F1=f1_score(y,p,average='macro'),acc=accuracy_score(y,p),FR=recall_score(y==3,p==3),FP=precision_score(y==3,p==3,zero_division=0),QWK=cohen_kappa_score(y,p,weights='quadratic'),경보=int((p==3).sum()),놓침=int(((y==3)&(p!=3)).sum()))
rows=[]
for name,cols in [('As-Is 계획 Top5',TOP5),('As-Is 계획 7개',BASE),('1단계 전체 19개',ALL),('2단계 Top8',TOP8),('3단계 Top8+파생',TOP8+DER),('3단계 전체+파생',ALL+DER)]:
    for a in ['RF','LR']:
        mv=Pipeline([('p',pre(cols)),('m',clf(a))]).fit(tr[cols],tr.y); rv=met(va.y.values,mv.predict(va[cols]))
        mb=Pipeline([('p',pre(cols)),('m',clf(a))]).fit(o[cols],o.y); rb=met(b.y.values,mb.predict(b[cols]))
        rows.append({'구성':name,'모델':a,'내부F1':rv['F1'],**{'Blind_'+k:v for k,v in rb.items()}})
r=pd.DataFrame(rows); pd.set_option('display.width',200); print(r.round(3).to_string(index=False))
r.round(4).to_csv(sys.argv[4],index=False,encoding='utf-8-sig')
