import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
d=pd.read_csv(sys.argv[1],encoding='utf-8-sig'); out=sys.argv[2]
d['delay_grade']=np.select([d.schedule_delay_min<18,d.schedule_delay_min<35,d.schedule_delay_min<60],[0,1,2],3)
d['on_time']=(d.on_time_completion_flag=='Y').astype(int)
for c in ['shift','line_id','product_family','order_priority','bottleneck_station']: pass
num=d.select_dtypes('number').columns.tolist()
corr=d[num].corr(method='spearman')
fig,ax=plt.subplots(figsize=(14,12)); im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(len(num)),num,rotation=90,fontsize=8); ax.set_yticks(range(len(num)),num,fontsize=8)
for i in range(len(num)):
    for j in range(len(num)): ax.text(j,i,f'{corr.iat[i,j]:.2f}',ha='center',va='center',fontsize=5.5,color='white' if abs(corr.iat[i,j])>0.5 else 'black')
fig.colorbar(im,fraction=0.04); ax.set_title(f'원본 1,600건 전체 수치 피처 Spearman 상관 ({len(num)}개, 전처리 전)'); plt.tight_layout(); plt.savefig(out+'/full_corr_heatmap.png',dpi=160)
print('heatmap cols',len(num))
print(corr['schedule_delay_min'].drop('schedule_delay_min').sort_values(key=abs,ascending=False).round(3).to_string())
S=["station_a_cycle_sec","station_b_cycle_sec","station_c_cycle_sec","station_d_cycle_sec","wait_before_b_min","wait_before_c_min","wait_before_d_min","wip_start_qty","wip_end_qty","changeover_min","equipment_stop_min","material_wait_min","planned_qty","actual_qty","available_operators","planned_lead_time_min","actual_lead_time_min","schedule_delay_min"]
t=d[S].describe(percentiles=[.05,.5,.95]).T[['min','5%','50%','mean','95%','max']]
t['0 비율']=(d[S]==0).mean(); t['결측']=d[S].isna().sum()
t.round(2).to_csv(out+'/sensor_minmax.csv',encoding='utf-8-sig'); print(t.round(2).to_string())
print(d.groupby('line_id')[['station_a_cycle_sec','station_b_cycle_sec','station_c_cycle_sec','station_d_cycle_sec']].mean().round(1))
print(pd.crosstab(d.line_id,d.bottleneck_station,normalize='index').round(2))
print(d.product_family.value_counts().to_dict(), d.shift.value_counts().to_dict())
