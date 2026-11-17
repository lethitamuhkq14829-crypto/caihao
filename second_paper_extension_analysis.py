from pathlib import Path
import json
import numpy as np
import pandas as pd
import second_paper_complete_analysis as core

ROOT=core.ROOT;PREV=core.OUT
OUT=ROOT/'output/tables/SecondPaper_Extension_20261109'
REPORT=ROOT/'output/reports/SecondPaper_Extension_20261109'
for p in [OUT,REPORT,OUT/'pairs']:p.mkdir(parents=True,exist_ok=True)
core.OUT=OUT;core.REPORT=REPORT
BANDS=[(0,30),(30,60),(60,90),(90,150),(150,300),(300,500)]
PROTO=dict(distance_bands_m=BANDS,closure='first [0,30], subsequent (lower,upper]',reference_m=[[500,2000],[1000,2000]],
 segment_lengths_m=[500,1000,2000],calipers=core.CONFIG['calipers'],exact_lc=True,
 PFR_initial_bins=[0,10,50,90,100],PFR_reporting_bins=['0','0<PFR<=50','50<PFR<=100'],
 PFR_bin_reason='Baseline support audit: Russia 0-10 has only one native cell; China >50 only one cell. Combine positive <=50 and flag >50 cross-country inference unsupported.',
 min_PFR_cells_for_interval=5,min_blocks_for_interval=5,seed=20261109,bootstrap_repetitions=2000,
 seasonal_status='Requires new raw-image composites; no local monthly source or Earth Engine credentials found.',
 legacy_status='Requires 2010-2014 Landsat7; no local pre-2013 NDVI source found.')
(REPORT/'extension_protocol.json').write_text(json.dumps(PROTO,indent=2),encoding='utf8')

def csv(f,name):core.csv(f,name)

def weighted_balance(f,t,c,label):
    rows=[]
    for route in ['Russia','China']:
      for typ in ['All','Forest','Cropland','Other vegetation']:
        m=f.route.to_numpy()[t]==route
        if typ!='All':m &= f.type.to_numpy()[t]==typ
        tt=t[m];cc=c[m]
        if len(tt)<2:continue
        u=f.unit.to_numpy()[tt];_,inv,count=np.unique(u,return_inverse=True,return_counts=True);w=1/count[inv];w=w/w.sum()
        for col in core.COV:
            x=f[col].to_numpy()[tt];y=f[col].to_numpy()[cc];mx=(w*x).sum();my=(w*y.mean(axis=1)).sum()
            sd=np.sqrt(((w*(x-mx)**2).sum()+(w[:,None]*(y-my)**2).sum()/3)/2)
            rows.append(dict(analysis=label,route=route,type=typ,variable=col,SMD=(mx-my)/sd if sd else 0,n=len(tt)))
    return rows

def distance_and_segments():
    f=core.build_inputs(max_target_m=500);summaries=[];ret=[];balances=[];segs=[]
    for ref in [(500,2000),(1000,2000)]:
      for lo,hi in BANDS:
        name=f'distance_{lo}_{hi}_ref{ref[0]}'
        t,c=core.match(f,name,inner=lo,near=hi,ref=ref)
        a=core.trajectories(f,t,c);keep=a.complete.to_numpy();a=a[keep].copy()
        summaries+=core.block_summary(a,name);balances+=weighted_balance(f,t[keep],c[keep],name)
        a['operation']=a[[f'D_{y}' for y in range(2020,2024)]].mean(axis=1)
        for route,g in a.groupby('route'):ret.append(dict(analysis=name,route=route,matched=int((f.route.to_numpy()[t]==route).sum()),complete=len(g),units=g.unit.nunique()))
        g=a.groupby(['route','unit','type']).agg(n=('pixel_id','size'),operation=('operation','mean')).reset_index();g['analysis']=name;segs.append(g)
        annual=a.groupby(['route','unit','type']).agg(n=('pixel_id','size'),**{f'D_{y}':(f'D_{y}','mean') for y in core.YEARS}).reset_index()
        csv(annual,f'{name}_segment_annual.csv')
    for size in [500,1000,2000]:
        f['unit']=(f.chainage_m/size).astype(int)
        name=f'segment_{size}m';t,c=core.match(f,name)
        a=core.trajectories(f,t,c);keep=a.complete.to_numpy();balances+=weighted_balance(f,t[keep],c[keep],name)
        a=a[keep].copy();a['unit']=a.unit*size/1000
        summaries+=core.block_summary(a,name)
        for route,g in a.groupby('route'):ret.append(dict(analysis=name,route=route,matched=int((f.route.to_numpy()[t]==route).sum()),complete=len(g),units=g.unit.nunique()))
    csv(pd.DataFrame(summaries),'extension_effects.csv');csv(pd.DataFrame(balances),'extension_balance.csv')
    csv(pd.DataFrame(ret),'extension_retention.csv');csv(pd.concat(segs),'distance_segment_effects.csv')
    return f

def pfr_analysis():
    a=pd.read_csv(PREV/'pairs/PFR_annual_targets.csv.gz');a=a[a.complete].copy()
    a['PFR_group']=np.where(a.PFR_pre==0,'0',np.where(a.PFR_pre<=50,'0-50','50-100'))
    allrows=[];counts=[]
    for layer in ['0','0-50','50-100']:
        x=a[a.PFR_group==layer];r=core.block_summary(x,'PFR_'+layer)
        for v in r:
            if v['n_PFR_cells']<5:v['ci_low']=np.nan;v['ci_high']=np.nan
            v['adequate_support']=bool(v['n_PFR_cells']>=5 and v['n_blocks']>=5)
        allrows+=r
        for (country,typ),g in x.groupby(['route','type']):
            counts.append(dict(route=country,type=typ,PFR_group=layer,n=len(g),cells=g.PFR_cell_id.nunique(),units=g.unit.nunique(),blocks10=(g.unit//10).nunique(),NDVI_pre=g.NDVI_pre.mean(),elevation=g.elevation.mean(),slope=g.slope.mean()))
    csv(pd.DataFrame(allrows),'PFR_stratified_effects.csv');csv(pd.DataFrame(counts),'PFR_stratum_support.csv')
    from second_paper_crosscountry import cross_stats
    pairs=pd.read_csv(PREV/'common_strict_PFR_pairs.csv.gz');pairs=pairs[pairs.complete]
    vals=a.set_index('pixel_id');r=vals.loc[pairs.Russia_target_id].reset_index();c=vals.loc[pairs.China_target_id].reset_index()
    results=[];bal=[];sup=[]
    for layer in ['0','0-50','50-100']:
        keep=(r.PFR_group==layer)&(c.PFR_group==layer)
        if keep.sum()<2:continue
        rr,bb=cross_stats(r[keep],c[keep],'strict_cross_PFR_'+layer)
        for v in rr:
            if min(v['Russia_PFR_cells'],v['China_PFR_cells'])<5:v['ci_low']=np.nan;v['ci_high']=np.nan
        results+=rr;bal+=bb
        sup.append(dict(PFR_group=layer,pairs=int(keep.sum()),Russia_cells=r[keep].PFR_cell_id.nunique(),China_cells=c[keep].PFR_cell_id.nunique()))
    csv(pd.DataFrame(results),'PFR_within_stratum_country_contrasts.csv');csv(pd.DataFrame(bal),'PFR_within_stratum_country_balance.csv');csv(pd.DataFrame(sup),'PFR_within_stratum_country_support.csv')
    modelrows=[];coefrows=[];modeldata=[]
    for route in ['Russia','China']:
      for typ in ['All','Forest','Cropland']:
        x=a[(a.route==route)&(a.PFR_pre<=50)].copy()
        if typ!='All':x=x[x.type==typ]
        x['operation']=x[[f'D_{y}' for y in range(2020,2024)]].mean(axis=1)
        g=x.groupby(['unit','PFR_cell_id','lc2013']).agg(n=('pixel_id','size'),operation=('operation','mean'),PFR_pre=('PFR_pre','mean'),NDVI_pre=('NDVI_pre','mean'),elevation=('elevation','mean'),slope=('slope','mean'),latitude=('latitude','mean')).reset_index()
        g['weight']=g.n/g.groupby('unit').n.transform('sum');g['route']=route;g['type']=typ;modeldata.append(g)
        X=np.column_stack([np.ones(len(g)),g.PFR_pre/10,g.NDVI_pre/.1,g.elevation/100,g.slope/5,g.latitude-g.latitude.mean(),pd.get_dummies(g.lc2013,drop_first=True,dtype=float).to_numpy()])
        columns=['intercept','PFR_per10pct','NDVI_pre_per0.1','elevation_per100m','slope_per5deg','latitude_per1deg']+[f'lc_{v}' for v in sorted(g.lc2013.unique())[1:]]
        y=g.operation.to_numpy();w=g.weight.to_numpy();rank=np.linalg.matrix_rank(X)
        if rank<X.shape[1]:
            modelrows.append(dict(route=route,type=typ,status='rank_deficient',n=len(g),rank=rank,columns=X.shape[1]));continue
        fit=lambda ww:np.linalg.lstsq(X*np.sqrt(ww)[:,None],y*np.sqrt(ww),rcond=None)[0]
        beta=fit(w);res=y-X@beta
        for j,col in enumerate(columns):coefrows.append(dict(route=route,type=typ,term=col,coefficient=beta[j]))
        for L in [5,10,20]:
            bid,inv=np.unique(g.unit.to_numpy()//L,return_inverse=True);rng=np.random.default_rng(20261109+L);boots=[]
            z=X[:,1:6];groups=g.lc2013.to_numpy()
            def slope_within_class(ww):
                zz=z.copy();yy=y.copy()
                for cl in np.unique(groups):
                    sel=groups==cl;total=ww[sel].sum()
                    if total>0:
                        zz[sel]-=np.average(z[sel],axis=0,weights=ww[sel])
                        yy[sel]-=np.average(y[sel],weights=ww[sel])
                design=zz*np.sqrt(ww)[:,None]
                if np.linalg.matrix_rank(design)<design.shape[1]:return np.nan
                return np.linalg.lstsq(design,yy*np.sqrt(ww),rcond=None)[0][0]
            assert abs(slope_within_class(w)-beta[1])<1e-9
            for k in range(2000):
                cnt=rng.multinomial(len(bid),np.full(len(bid),1/len(bid)));ww=w*cnt[inv]
                value=slope_within_class(ww)
                if np.isfinite(value):boots.append(value)
            interval=np.quantile(boots,[.025,.975]) if len(boots)>=1800 else [np.nan,np.nan]
            modelrows.append(dict(route=route,type=typ,status='ok' if len(boots)>=1800 else 'bootstrap_rank_unstable',n=len(g),native_cells=g.PFR_cell_id.nunique(),units=g.unit.nunique(),blocks=len(bid),block_km=L,coefficient_PFR_per10pct=beta[1],ci_low=interval[0],ci_high=interval[1],valid_bootstraps=len(boots),weighted_R2=1-np.sum(w*res**2)/np.sum(w*(y-np.average(y,weights=w))**2)))
    csv(pd.DataFrame(modelrows),'PFR_conditional_association.csv');csv(pd.DataFrame(coefrows),'PFR_model_coefficients.csv');csv(pd.concat(modeldata),'PFR_model_input_cells.csv')
    s=pd.read_csv(PREV/'base_segment_annual.csv');s['operation']=s[[f'D_{y}' for y in range(2020,2024)]].mean(axis=1)
    acf=[]
    for route,g in s.groupby('route'):
        raw=g.set_index('unit').operation.reindex(range(int(g.unit.max())+1))
        xx=np.arange(len(raw));ok=raw.notna().to_numpy();trend=np.polyval(np.polyfit(xx[ok],raw.to_numpy()[ok],2),xx)
        for name,series in [('raw',raw),('quadratic_detrended',raw-trend)]:
            for lag in range(1,31):
                x=series.to_numpy()[:-lag];y=series.to_numpy()[lag:];good=np.isfinite(x)&np.isfinite(y)
                acf.append(dict(route=route,series=name,lag_km=lag,pairs=int(good.sum()),correlation=float(np.corrcoef(x[good],y[good])[0,1])))
    csv(pd.DataFrame(acf),'along_pipeline_autocorrelation.csv')
    print('PFR AND SPATIAL CHECKS COMPLETE',flush=True)

if __name__=='__main__':
    distance_and_segments();pfr_analysis();print('EXTENSION ANALYSIS COMPLETE',flush=True)
