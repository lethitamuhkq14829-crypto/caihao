import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from second_paper_complete_analysis import OUT,REPORT,COV,CAL,YEARS,CONFIG,csv,trajectories,balance

def cross_pair(f,t,c,pfr=False,strict=False):
    z=f.iloc[t].copy();z['local_pair_index']=np.arange(len(t))
    if pfr:z=z[z.PFR_pre_valid]
    basecal=np.array([.01,5.,1.]) if strict else CAL
    cov=COV+(['PFR_pre'] if pfr else []);cal=np.r_[basecal,5. if strict else 10.] if pfr else basecal
    pairs=[];audit=[]
    for lc,g in z.groupby('lc2013'):
        ru=g[g.route=='Russia'];cn=g[g.route=='China']
        if len(ru)==0 or len(cn)==0:continue
        rv=ru[cov].to_numpy()/cal;cv=cn[cov].to_numpy()/cal
        opts=cKDTree(cv).query_ball_point(rv,1+1e-10,p=np.inf)
        order=np.lexsort((ru.pixel_id.to_numpy(),np.array([len(o) for o in opts])))
        used=set();n=0
        for i in order:
            avail=np.array([j for j in opts[i] if j not in used],dtype=int)
            if not len(avail):continue
            d=((cv[avail]-rv[i])**2).sum(axis=1)
            j=avail[np.lexsort((cn.pixel_id.to_numpy()[avail],d))[0]]
            used.add(int(j));n+=1
            pairs.append((int(ru.local_pair_index.iloc[i]),int(cn.local_pair_index.iloc[j])))
        audit.append(dict(lc2013=int(lc),Russia_candidates=len(ru),China_candidates=len(cn),pairs=n))
    return np.asarray(pairs,dtype=int).reshape(-1,2),pd.DataFrame(audit)

def cross_stats(r,c,label,reps=2000):
    metrics=[f'D_{y}' for y in YEARS]+['operation','operation_2020_2022','rebound_2023_minus_2019','baseline_drift']
    rows=[];balances=[]
    for typ in ['All','Forest','Cropland','Other vegetation']:
        mask=np.ones(len(r),bool) if typ=='All' else r.type.to_numpy()==typ
        a=r[mask].reset_index(drop=True).copy();b=c[mask].reset_index(drop=True).copy()
        if len(a)<2:continue
        for v in [a,b]:
            v['operation']=v[[f'D_{y}' for y in range(2020,2024)]].mean(axis=1)
            v['operation_2020_2022']=v[[f'D_{y}' for y in range(2020,2023)]].mean(axis=1)
            v['rebound_2023_minus_2019']=v.D_2023-v.D_2019
            v['baseline_drift']=v.D_2014-v.D_2013
        weights=1/a.groupby('unit').pixel_id.transform('count').to_numpy();weights/=weights.sum()
        for v in COV+['PFR_pre']:
            x=a[v].to_numpy();y=b[v].to_numpy();ok=np.isfinite(x)&np.isfinite(y)
            if ok.sum()<2:continue
            w=weights[ok]/weights[ok].sum();mx=np.sum(w*x[ok]);my=np.sum(w*y[ok])
            sd=np.sqrt((np.sum(w*(x[ok]-mx)**2)+np.sum(w*(y[ok]-my)**2))/2)
            balances.append(dict(analysis=label,type=typ,covariate=v,Russia_mean=mx,China_mean=my,SMD=(mx-my)/sd if sd else 0,n_pairs=len(a)))
        for L in [5,10,20]:
            ru,ri=np.unique(a.unit.to_numpy()//L,return_inverse=True);cn,ci=np.unique(b.unit.to_numpy()//L,return_inverse=True)
            W=np.zeros((len(ru),len(cn)));np.add.at(W,(ri,ci),weights)
            X=np.zeros((len(ru),len(cn),len(metrics),3))
            val=np.stack([a[metrics].to_numpy(),b[metrics].to_numpy(),a[metrics].to_numpy()-b[metrics].to_numpy()],axis=-1)
            np.add.at(X,(ri,ci),weights[:,None,None]*val)
            rng=np.random.default_rng(CONFIG['seed']+L)
            rw=rng.multinomial(len(ru),np.full(len(ru),1/len(ru)),reps)
            cw=rng.multinomial(len(cn),np.full(len(cn),1/len(cn)),reps)
            den=np.einsum('bi,ij,bj->b',rw,W,cw)
            num=np.einsum('bi,ijmk,bj->bmk',rw,X,cw,optimize=True)
            boot=num[den>0]/den[den>0,None,None];lo,hi=np.quantile(boot,[.025,.975],axis=0)
            means=np.sum(weights[:,None,None]*val,axis=0)
            for j,m in enumerate(metrics):
                for k,est in enumerate(['Russia','China','Russia_minus_China']):
                    rows.append(dict(analysis=label,type=typ,metric=m,estimate=est,mean=means[j,k],ci_low=lo[j,k] if min(len(ru),len(cn))>=5 else np.nan,ci_high=hi[j,k] if min(len(ru),len(cn))>=5 else np.nan,block_km=L,n_pairs=len(a),Russia_units=a.unit.nunique(),China_units=b.unit.nunique(),Russia_blocks=len(ru),China_blocks=len(cn),Russia_PFR_cells=a.loc[a.PFR_pre.notna(),'PFR_cell_id'].nunique(),China_PFR_cells=b.loc[b.PFR_pre.notna(),'PFR_cell_id'].nunique()))
    return rows,balances

def main():
    f=pd.read_pickle(OUT/'analysis_inputs.pkl');results=[];bal=[];ret=[]
    for local,usepfr,strict in [('base',False,False),('PFR',True,False),('base',False,True),('PFR',True,True)]:
        z=np.load(OUT/'pairs'/f'{local}.npz');t=z['target'];c=z['controls'];name=('common_strict_' if strict else 'common_')+local
        pair,audit=cross_pair(f,t,c,usepfr,strict);csv(audit,name+'_support_by_lc.csv')
        annual=trajectories(f,t,c)
        r=annual.iloc[pair[:,0]].reset_index(drop=True);ch=annual.iloc[pair[:,1]].reset_index(drop=True)
        good=r.complete.to_numpy()&ch.complete.to_numpy()
        table=pd.DataFrame(dict(Russia_target_id=r.pixel_id,China_target_id=ch.pixel_id,Russia_unit=r.unit,China_unit=ch.unit,lc2013=r.lc2013,complete=good))
        assert table.Russia_target_id.is_unique and table.China_target_id.is_unique and (r.lc2013==ch.lc2013).all()
        bc=np.array([.01,5.,1.]) if strict else CAL
        for v,cal in zip(COV+(['PFR_pre'] if usepfr else []),np.r_[bc,5. if strict else 10.] if usepfr else bc):assert np.max(np.abs(r[v]-ch[v]))<=cal+1e-6
        csv(table,name+'_pairs.csv.gz')
        _,b=cross_stats(r,ch,name+'_baseline',reps=10);bal+=b
        rows,b=cross_stats(r[good],ch[good],name);results+=rows;bal+=b
        ret.append(dict(analysis=name,baseline_pairs=len(pair),complete_pairs=int(good.sum()),Russia_baseline_targets=int((annual.route=='Russia').sum()),China_baseline_targets=int((annual.route=='China').sum()),Russia_units_complete=r[good].unit.nunique(),China_units_complete=ch[good].unit.nunique()))
        if local=='base':
            selected_r=r[good].copy();selected_c=ch[good].copy()
            selected_r['country_pair_id']=np.arange(len(selected_r));selected_c['country_pair_id']=np.arange(len(selected_c))
            csv(pd.concat([selected_r,selected_c]),name+'_complete_annual.csv.gz')
        print('COMMON SUPPORT',name,'baseline',len(pair),'complete',good.sum(),flush=True)
    csv(pd.DataFrame(results),'crosscountry_effects.csv');csv(pd.DataFrame(bal),'crosscountry_balance.csv');csv(pd.DataFrame(ret),'crosscountry_retention.csv')

if __name__=='__main__':main()
