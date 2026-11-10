import sys, pandas as pd, numpy as np
d=pd.read_csv(sys.argv[1],encoding='utf-8-sig',parse_dates=['lot_start_time'])
d['y']=np.select([d.schedule_delay_min<18,d.schedule_delay_min<35,d.schedule_delay_min<60],[0,1,2],3)
d=d.sort_values(['lot_start_time','record_id']).reset_index(drop=True)
g=d.groupby('line_id')
d['prev_fam']=g.product_family.shift()
d['fam_change']=(d.product_family!=d.prev_fam)&d.prev_fam.notna()
d['gap_min']=(d.lot_start_time-g.lot_start_time.shift()).dt.total_seconds()/60
d['hour']=d.lot_start_time.dt.hour
d['dow']=d.lot_start_time.dt.dayofweek
print('lines',d.line_id.value_counts().to_dict())
print(pd.crosstab(d.fam_change,d.changeover_min>0))
print(d.groupby('fam_change').y.agg(['mean','count']), d.groupby('fam_change').apply(lambda x:(x.y==3).mean()))
print(d.groupby(pd.qcut(d.gap_min,5,duplicates='drop')).apply(lambda x:pd.Series({'n':len(x),'fail':(x.y==3).mean(),'ymean':x.y.mean()})))
print(d.groupby('hour').apply(lambda x:pd.Series({'n':len(x),'fail':(x.y==3).mean()})).T.round(2))
print(d.groupby('dow').apply(lambda x:(x.y==3).mean()).round(3).to_dict())
print(d.groupby('order_priority').apply(lambda x:(x.y==3).mean()).round(3).to_dict())
# autocorrelation: prev lot delay on same line
d['prev_delay']=g.schedule_delay_min.shift()
print('corr prev delay',d[['prev_delay','schedule_delay_min']].corr().iloc[0,1])
print('corr material/stop/changeover with delay',d[['material_wait_min','equipment_stop_min','changeover_min','schedule_delay_min']].corr()['schedule_delay_min'].round(3).to_dict())
d['week']=d.lot_start_time.dt.isocalendar().week
print(d.groupby('week').apply(lambda x:(x.y==3).mean()).round(2).to_dict())
