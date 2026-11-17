"""Auditable local baseline matching and longitudinal analysis. Raw files read-only.
Run with python -X utf8 scripts/second_paper_complete_analysis.py
All distances and chainages are in EPSG:32652. No outcome-dependent matching.
"""
from pathlib import Path
from contextlib import ExitStack
import json, hashlib, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import shapely
from shapely.geometry import LineString
from shapely.affinity import translate
from scipy.spatial import cKDTree
from pyproj import Transformer
import geopandas as gpd
import analyze_crossborder_gee_ndvi_20261106 as old
from build_second_paper_baseline_samples import NativePFR, country_geometries, slope_horn, VEGETATION

ROOT=Path(__file__).resolve().parent
SOURCE=(Path(__file__).resolve().parent / 'GEE_downloads')
OUT=ROOT/'output/tables/SecondPaper_Complete_20261109'
REPORT=ROOT/'output/reports/SecondPaper_Complete_20261109'
YEARS=list(range(2013,2024)); NC=[f'NDVI_{y}' for y in YEARS]; OC=[f'NOBS_{y}' for y in YEARS]
COV=['NDVI_pre','elevation','slope']; CAL=np.array([.05,50.,5.])
CONFIG=dict(seed=20261109,near_m=50,reference_m=[500,2000],controls=3,
 calipers=dict(zip(COV,CAL)),PFR_caliper_percentage_points=10,exact=['country','1km segment','lc2013'],
 replacement='between targets only; three distinct controls per target',
 annual_cohort='all four members have NOBS>=1 in every 2013-2023 year',
 bootstrap='fixed non-overlapping consecutive chainage blocks, resampled with replacement; ratio of summed segment values to summed segment counts',
 bootstrap_repetitions=2000,bootstrap_block_km=[5,10,20],
 shifts_m=[[30,0],[-30,0],[0,30],[0,-30]],
 sensitivity=['near30','near100','ring50_100','ref500_1000','ref1000_2000','border3km_excluded','lc_stable','adjacent1km','PFR','NOBS3','end2022','line_shifts'])
for p in [OUT,REPORT,OUT/'pairs']:p.mkdir(parents=True,exist_ok=True)
config_path=REPORT/'analysis_protocol.json'
if config_path.exists():assert json.loads(config_path.read_text())==CONFIG
else:config_path.write_text(json.dumps(CONFIG,indent=2),encoding='utf8')

def csv(f,name):
    f.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.9g')

def line_index(parts):
    edges=[];starts=[];owners=[]
    for k,line in enumerate(parts):
        xy=list(line.coords);e=[LineString([a,b]) for a,b in zip(xy,xy[1:])]
        edges.extend(e);starts.extend(np.r_[0,np.cumsum(shapely.length(e))[:-1]]);owners.extend([k]*len(e))
    edges=np.array(edges,dtype=object)
    return edges,np.array(starts),np.array(owners),shapely.STRtree(edges)

def locate(x,y,index):
    e,starts,owners,tree=index;pts=shapely.points(x,y);ii=tree.nearest(pts)
    return shapely.distance(e[ii],pts),starts[ii]+shapely.line_locate_point(e[ii],pts),owners[ii]

def build_inputs(max_target_m=131):
    cache=OUT/'analysis_inputs.pkl'
    if cache.exists():return pd.read_pickle(cache)
    digest,counts=old.verify_logs(SOURCE/'第一步')
    full,parts=old.get_geometry();index=line_index(parts)
    bounds=gpd.GeoSeries([full.buffer(10000)],crs=32652).to_crs(4326).total_bounds
    pf=NativePFR(ROOT,bounds);countries,cs=country_geometries(ROOT,bounds)
    tr=Transformer.from_crs(32652,4326,always_xy=True)
    files={f'NDVI_{y}':SOURCE/f'第一步/CN_RU_NDVI_{y}_30m.tif' for y in YEARS}
    files.update({f'NOBS_{y}':SOURCE/f'第一步/CN_RU_NOBS_{y}_30m.tif' for y in YEARS})
    files.update(lc2013=SOURCE/'第二步/CN_RU_GLC_FCS30D_2013_30m.tif',lc2014=SOURCE/'第二步/CN_RU_GLC_FCS30D_2014_30m.tif',elevation=SOURCE/'第二步/CN_RU_SRTM_DEM_30m.tif')
    frames=[];raw=0;gates=[]
    # Covers every shifted 50-m target and 500-2000-m reference, plus nominal 100-m targets.
    roi=shapely.union_all([full.buffer(max_target_m+9,quad_segs=64),full.buffer(2040,quad_segs=64).difference(full.buffer(460,quad_segs=64))])
    with ExitStack() as stack:
        ds={k:stack.enter_context(rasterio.open(p)) for k,p in files.items()};base=ds[NC[0]]
        for s in ds.values():assert (s.crs,s.shape,s.transform)==(base.crs,base.shape,base.transform)
        assert base.crs.to_epsg()==32652 and base.res==(30,30)
        for start in range(0,base.height,256):
            w=Window(0,start,base.width,min(256,base.height-start));aff=base.window_transform(w)
            m=rasterize([(roi,1)],out_shape=(int(w.height),int(w.width)),transform=aff,dtype='uint8').astype(bool)
            m &= ds[OC[0]].read(1,window=w)!=65535
            rr,cc=np.nonzero(m)
            if not len(rr):continue
            x,y=aff*(cc+.5,rr+.5);dist,chain,owner=locate(x,y,index)
            keep=(dist<=max_target_m)|((dist>=469)&(dist<=2031))
            rr,cc,x,y,dist,chain,owner=[a[keep] for a in [rr,cc,x,y,dist,chain,owner]]
            cn=shapely.contains_xy(countries[0],x,y);ru=shapely.contains_xy(countries[1],x,y)
            country=np.where(cn&~ru,'China',np.where(ru&~cn,'Russia','Ambiguous'))
            route=np.array(old.SIDES)[owner]
            lon,lat=tr.transform(x,y);pfr,pfid,pfv=pf.sample(lon,lat)
            sl=slope_horn(ds['elevation'],w)[rr,cc]
            f=pd.DataFrame(dict(pixel_id=(rr+start).astype('int64')*base.width+cc,x=x,y=y,longitude=lon,latitude=lat,
                route=route,country=country,distance_m=dist,chainage_m=chain,unit=(chain//1000).astype(int),slope=sl,
                PFR_pre=pfr,PFR_cell_id=pfid,PFR_pre_valid=pfv,
                border_distance_m=shapely.distance(shapely.points(x,y),countries[0].boundary)))
            for k,s in ds.items():
                v=s.read(1,window=w)[rr,cc]
                if k.startswith('NDVI_') or k=='elevation':v=np.where(v==s.nodata,np.nan,v)
                f[k]=v
            f['NDVI_pre']=f[NC[:2]].mean(axis=1,skipna=False)
            f['landcover_baseline_same']=f.lc2013==f.lc2014
            valid=(f.country==f.route)&f.route.isin(['Russia','China'])&f.lc2013.isin(VEGETATION)&f[NC[:2]].notna().all(axis=1)&(f[OC[:2]]>=1).all(axis=1)&f.slope.between(0,90)&np.isfinite(f.elevation)
            raw+=len(f);frames.append(f[valid]);gates.append(dict(row=start,raw=len(f),eligible=int(valid.sum())))
            if start%2048==0:print('EXTRACT',start,'eligible so far',sum(len(v) for v in frames),flush=True)
    f=pd.concat(frames,ignore_index=True).sort_values('pixel_id').reset_index(drop=True)
    assert f.pixel_id.is_unique
    f['type']=np.where(f.lc2013.isin([10,11,12,20]),'Cropland',np.where(f.lc2013.between(51,92),'Forest','Other vegetation'))
    f.to_pickle(cache)
    csv(pd.DataFrame(gates),'extraction_gates.csv')
    csv(f.groupby(['route','type','PFR_pre_valid']).agg(n=('pixel_id','size'),NDVI_pre=('NDVI_pre','mean'),elevation=('elevation','mean'),slope=('slope','mean')).reset_index(),'PFR_coverage_profile.csv')
    manifest=dict(raw_extracted=raw,baseline_eligible=len(f),grid=str(base.crs),pipeline_sha256=digest,
        route_km=dict(zip(old.SIDES,[g.length/1000 for g in parts])),files={k:dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in files.items()},country_sources=cs)
    (REPORT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('EXTRACT DONE',len(f),flush=True)
    return f

def match(f,name,near=50,inner=0,ref=(500,2000),pfr=False,stable=False,border=False,adjacent=False,shift=(0,0)):
    path=OUT/'pairs'/f'{name}.npz'
    if path.exists():
        z=np.load(path);return z['target'],z['controls']
    if shift!=(0,0):
        _,parts=old.get_geometry();ix=line_index([translate(g,xoff=shift[0],yoff=shift[1]) for g in parts])
        dist,chain,owner=locate(f.x.to_numpy(),f.y.to_numpy(),ix)
        country_ok=np.array(old.SIDES)[owner]==f.route.to_numpy();unit=(chain//1000).astype(int)
    else:dist=f.distance_m.to_numpy();unit=f.unit.to_numpy();country_ok=np.ones(len(f),bool)
    valid=country_ok.copy()
    if pfr:valid &= f.PFR_pre_valid.to_numpy()
    if stable:valid &= f.landcover_baseline_same.to_numpy()
    if border:valid &= f.border_distance_m.to_numpy()>3000
    tm=valid&(dist<=near)&(dist>inner if inner else dist>=0);cm=valid&(dist>=ref[0])&(dist<=ref[1])
    cov=COV+(['PFR_pre'] if pfr else []);cal=np.r_[CAL,10.] if pfr else CAL
    vals=f[cov].to_numpy()/cal
    groups=pd.DataFrame(dict(route=f.route,unit=unit,lc=f.lc2013))
    cgroups=groups[cm].groupby(['route','unit','lc']).groups
    ts=[];cs=[];unmatched=[];coverage=[]
    for key,tix in groups[tm].groupby(['route','unit','lc']).groups.items():
        tix=np.array(tix);route,u,lc=key
        if adjacent:ci=np.concatenate([np.asarray(cgroups.get((route,u+du,lc),[]),dtype=int) for du in [-1,0,1]])
        else:ci=np.asarray(cgroups.get(key,[]),dtype=int)
        ci=np.sort(ci);matched=0
        if len(ci)>=3:
            tree=cKDTree(vals[ci]);options=tree.query_ball_point(vals[tix],r=1+1e-10,p=np.inf)
            for t,opts in zip(tix,options):
                if len(opts)<3:unmatched.append((int(t),'fewer_than_3_within_calipers'));continue
                idx=ci[np.asarray(opts)];d=((vals[idx]-vals[t])**2).sum(axis=1)
                chosen=idx[np.lexsort((f.pixel_id.to_numpy()[idx],d))[:3]]
                ts.append(t);cs.append(chosen);matched+=1
        else:unmatched.extend((int(t),'fewer_than_3_same_class_local_controls') for t in tix)
        coverage.append(dict(scenario=name,route=route,unit=int(u),lc2013=int(lc),targets=len(tix),matched=matched,references=len(ci)))
    t=np.asarray(ts,dtype=int);c=np.asarray(cs,dtype=int).reshape(-1,3)
    order=np.argsort(t);t,c=t[order],c[order]
    assert len(np.unique(t))==len(t) and np.all(c[:,1:]!=c[:,:1]) and np.all(c[:,1]!=c[:,2])
    assert (f.lc2013.to_numpy()[t,None]==f.lc2013.to_numpy()[c]).all()
    assert (f.route.to_numpy()[t,None]==f.route.to_numpy()[c]).all()
    assert np.max(np.abs(vals[t,None,:]-vals[c]),initial=0)<=1+1e-8
    np.savez_compressed(path,target=t,controls=c)
    pairs=pd.DataFrame(dict(target_id=f.pixel_id.to_numpy()[t],route=f.route.to_numpy()[t],unit=unit[t],lc2013=f.lc2013.to_numpy()[t]))
    for j in range(3):pairs[f'control{j+1}_id']=f.pixel_id.to_numpy()[c[:,j]]
    pairs['each_control_weight']=1/3;csv(pairs,'pairs/'+name+'.csv.gz')
    csv(pd.DataFrame(coverage),'pairs/'+name+'_coverage.csv')
    csv(pd.DataFrame([dict(pixel_id=int(f.pixel_id.iloc[i]),reason=r) for i,r in unmatched]),'pairs/'+name+'_unmatched.csv.gz')
    print('MATCH',name,'targets',tm.sum(),'matched',len(t),'unique controls',len(np.unique(c)),flush=True)
    return t,c

def balance(f,t,c,label):
    rows=[]
    for route in ['Russia','China']:
      for typ in ['All','Forest','Cropland','Other vegetation']:
        m=f.route.to_numpy()[t]==route
        if typ!='All':m &= f.type.to_numpy()[t]==typ
        a=t[m];b=c[m].ravel()
        if len(a)<2:continue
        for col in COV+['PFR_pre','longitude','latitude']:
            x=f[col].to_numpy()[a];y=f[col].to_numpy()[b]
            x=x[np.isfinite(x)];y=y[np.isfinite(y)]
            if len(x)<2 or len(y)<2:continue
            sd=np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2)
            rows.append(dict(analysis=label,route=route,type=typ,covariate=col,n_targets=len(a),mean_target=x.mean(),mean_control=y.mean(),SMD=(x.mean()-y.mean())/sd if sd>0 else 0,variance_ratio=np.var(x)/np.var(y) if np.var(y)>0 else np.nan))
    return rows

def trajectories(f,t,c,obsmin=1,end=2023):
    nd=f[NC].to_numpy();ob=f[OC].to_numpy()
    q=(ob[t]>=obsmin)&(ob[c]>=obsmin).all(axis=1)&np.isfinite(nd[t])&np.isfinite(nd[c]).all(axis=1)
    cohort=q[:,:end-2013+1].all(axis=1)
    gap=nd[t]-nd[c].mean(axis=1);d=gap-gap[:,:2].mean(axis=1)[:,None]
    frame=f.iloc[t][['pixel_id','route','unit','type','lc2013','NDVI_pre','elevation','slope','PFR_pre','PFR_cell_id','longitude','latitude','distance_m','chainage_m']].copy().reset_index(drop=True)
    frame['complete']=cohort
    for j,y in enumerate(YEARS):frame[f'D_{y}']=d[:,j];frame[f'valid_{y}']=q[:,j];frame[f'near_{y}']=nd[t,j];frame[f'ref_{y}']=nd[c][:,:,j].mean(axis=1)
    return frame

def block_summary(frame,label,blocks=(5,10,20),reps=2000):
    # Every target first contributes within its 1-km unit; units then have equal weights.
    outputs=[]
    for route in ['Russia','China']:
      for typ in ['All','Forest','Cropland','Other vegetation']:
        x=frame[frame.route==route]
        if typ!='All':x=x[x.type==typ]
        if x.empty:continue
        cols=[f'D_{y}' for y in YEARS]+['transition','operation','operation_2020_2022','rebound_2023_minus_2019','baseline_drift_2014_minus_2013']
        x=x.copy();x['transition']=x[[f'D_{y}' for y in range(2015,2020)]].mean(axis=1)
        x['operation']=x[[f'D_{y}' for y in range(2020,2024)]].mean(axis=1)
        x['operation_2020_2022']=x[[f'D_{y}' for y in range(2020,2023)]].mean(axis=1)
        x['rebound_2023_minus_2019']=x.D_2023-x.D_2019
        x['baseline_drift_2014_minus_2013']=x.D_2014-x.D_2013
        seg=x.groupby('unit')[cols].mean();values=seg.to_numpy()
        for L in blocks:
            keys=seg.index.to_numpy()//L;bid,inv=np.unique(keys,return_inverse=True)
            sums=np.zeros((len(bid),len(cols)));np.add.at(sums,inv,values)
            counts=np.bincount(inv);rng=np.random.default_rng(CONFIG['seed']+L)
            picks=rng.integers(0,len(bid),(reps,len(bid)))
            boot=sums[picks].sum(axis=1)/counts[picks].sum(axis=1)[:,None]
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            for j,col in enumerate(cols):outputs.append(dict(analysis=label,route=route,type=typ,metric=col,mean=values[:,j].mean(),ci_low=lo[j] if len(bid)>=5 else np.nan,ci_high=hi[j] if len(bid)>=5 else np.nan,block_km=L,n_blocks=len(bid),n_units=len(seg),n_targets=len(x),n_PFR_cells=x.loc[x.PFR_pre.notna(),'PFR_cell_id'].nunique()))
    return outputs

def main():
    f=build_inputs(); print('INPUTS',len(f),flush=True)
    scenarios=[('base',{}),('PFR',{'pfr':True}),('near30',{'near':30}),('near100',{'near':100}),('ring50_100',{'near':100,'inner':50}),('ref500_1000',{'ref':(500,1000)}),('ref1000_2000',{'ref':(1000,2000)}),('border3km_excluded',{'border':True}),('lc_stable',{'stable':True}),('adjacent1km',{'adjacent':True})]
    scenarios += [(n,{'shift':s}) for n,s in [('shift_E30',(30,0)),('shift_W30',(-30,0)),('shift_N30',(0,30)),('shift_S30',(0,-30))]]
    balances=[];summaries=[];retention=[];coverage=[]
    for name,kw in scenarios:
        t,c=match(f,name,**kw);balances+=balance(f,t,c,name+'_baseline')
        a=trajectories(f,t,c);complete=a[a.complete].copy()
        balances+=balance(f,t[a.complete.to_numpy()],c[a.complete.to_numpy()],name+'_complete')
        summaries+=block_summary(complete,name)
        csv(a[[v for v in a.columns if not v.startswith(('near_','ref_'))]],'pairs/'+name+'_annual_targets.csv.gz')
        for route in ['Russia','China']:
            b=a[a.route==route];cc=complete[complete.route==route]
            retention.append(dict(analysis=name,route=route,matched_baseline=len(b),complete_targets=len(cc),analysis_end_year=2023,complete_units=cc.unit.nunique(),PFR_missing_targets=int(b.PFR_pre.isna().sum())))
            for y in YEARS:coverage.append(dict(analysis=name,route=route,year=y,matched=len(b),valid=int(b[f'valid_{y}'].sum())))
        if name=='base':
            a.to_pickle(OUT/'base_annual_targets.pkl')
            csv(complete.groupby(['route','unit']).agg(n=('pixel_id','size'),**{f'D_{y}':(f'D_{y}','mean') for y in YEARS},longitude=('longitude','mean'),latitude=('latitude','mean')).reset_index(),'base_segment_annual.csv')
            csv(complete,'base_complete_targets.csv.gz')
            for th,end,n in [(3,2023,'NOBS3'),(1,2022,'end2022')]:
                aa=trajectories(f,t,c,th,end);sel=aa.complete.to_numpy()
                if end==2022:
                    # 2023 values may be missing in this additional cohort. No 2023 or four-year operation estimate.
                    aa.loc[:,['D_2023']]=np.nan
                results=block_summary(aa[sel],n)
                if end==2022:results=[r for r in results if r['metric'] not in ['D_2023','operation','rebound_2023_minus_2019']]
                summaries+=results;balances+=balance(f,t[sel],c[sel],n+'_complete')
                for route in ['Russia','China']:
                    m=(aa.route==route)&sel;retention.append(dict(analysis=n,route=route,matched_baseline=int((aa.route==route).sum()),complete_targets=int(m.sum()),analysis_end_year=end,complete_units=aa[m].unit.nunique(),PFR_missing_targets=int(aa[m].PFR_pre.isna().sum())))
            # Remove PFR-missing targets without modifying the existing base controls.
            summaries+=block_summary(complete[complete.PFR_pre.notna()],'base_PFRvalid_targets')
    # Same target cohort: PFR-adjusted matching vs base matching; require complete members in both.
    base=pd.read_pickle(OUT/'base_annual_targets.pkl');pfr=pd.read_csv(OUT/'pairs/PFR_annual_targets.csv.gz')
    common=np.intersect1d(base.loc[base.complete,'pixel_id'],pfr.loc[pfr.complete,'pixel_id'])
    summaries+=block_summary(base[base.pixel_id.isin(common)],'base_same_PFR_cohort')
    summaries+=block_summary(pfr[pfr.pixel_id.isin(common)],'PFR_same_cohort')
    csv(pd.DataFrame(summaries),'effect_summaries.csv');csv(pd.DataFrame(balances),'balance.csv')
    csv(pd.DataFrame(retention),'retention.csv');csv(pd.DataFrame(coverage),'annual_coverage.csv')
    # Before-matching descriptive balance among all nominal near/reference candidates, by country.
    rows=[]
    for route in ['Russia','China']:
      for typ in ['All','Forest','Cropland','Other vegetation']:
        g=f[f.route==route]
        if typ!='All':g=g[g.type==typ]
        x=g[g.distance_m<=50];y=g[g.distance_m.between(500,2000)]
        for col in COV+['PFR_pre']:
            xx=x[col].dropna();yy=y[col].dropna();sd=np.sqrt((xx.var()+yy.var())/2)
            rows.append(dict(route=route,type=typ,covariate=col,SMD=(xx.mean()-yy.mean())/sd,n_near=len(x),n_ref=len(y)))
    csv(pd.DataFrame(rows),'balance_unmatched.csv')
    print('ANALYSIS CORE DONE',flush=True)

if __name__=='__main__':main()
