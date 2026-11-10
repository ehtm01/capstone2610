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
def run(name,cols,alg):
    m=Pipeline([('p',pre(cols)),('m',clf(alg))]).fit(tr[cols],tr.y); p=m.predict(va[cols]); rows.append((name,met(va.y.values,p)))
for e in EVENT: run('RF Top5 + '+e,TOP5+[e],'RF')
run('RF Top5 + changeover+material',TOP5+['changeover_min','material_wait_min'],'RF')
r=pd.DataFrame({k:v for k,v in rows}).T; print(r.round(3).to_string())
