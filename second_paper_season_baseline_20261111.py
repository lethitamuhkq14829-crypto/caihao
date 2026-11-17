from pathlib import Path
import json,re,zipfile,hashlib
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import shapely
from shapely.geometry import box
import second_paper_complete_analysis as core
from second_paper_crosscountry import cross_stats

ROOT=core.ROOT;PREV=core.OUT;SRC=(Path(__file__).resolve().parent / 'GEE_downloads')
OUT=ROOT/'output/tables/SecondPaper_SeasonBaseline_20261111'
REP=ROOT/'output/reports/SecondPaper_SeasonBaseline_20261111'
DATA=ROOT/'data/second_paper/supplement_ingest_20261111'
for p in [OUT,REP,DATA]:p.mkdir(exist_ok=True,parents=True)
PAT=re.compile(r'CN_RU_(L8_JulAug|L7_JunSep)_(NDVI|NOBS)_(\d{4})_30m-(\d+)-(\d+)')
def csv(f,name):f.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.10g')
def write(name,obj):(REP/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def ingest():
    folders=[SRC/'7—8 月 NDVI＋NOBS',SRC/'2010—2014 年早期基线']
    archives=[]
    for zpath in folders[1].rglob('*.zip'):
        with zipfile.ZipFile(zpath) as z:
            assert z.testzip() is None
            for item in z.infolist():
                if item.is_dir():continue
                rel=Path(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
                dest=DATA/'unzipped'/rel;dest.parent.mkdir(exist_ok=True,parents=True)
                content=z.read(item)
                if dest.exists():assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(content).digest()
                else:dest.write_bytes(content)
            archives.append(dict(path=str(zpath),bytes=zpath.stat().st_size,members=len(z.infolist()),CRC='passed'))
    roots=folders+[DATA/'unzipped'];paths=[p for r in roots for p in r.rglob('*') if p.is_file()]
    logs=pd.concat([pd.read_csv(p) for p in paths if p.suffix=='.csv' and '_log_' in p.name],ignore_index=True)
    scenes=pd.concat([pd.read_csv(p) for p in paths if p.suffix=='.csv' and '_scenes_' in p.name],ignore_index=True)
    assert not logs.duplicated(['tag','year']).any()
    for tag,ys in [('L8_JulAug',range(2013,2024)),('L7_JunSep',range(2010,2015))]:
        log=logs[logs.tag==tag];assert set(log.year)==set(ys)
        for _,v in log.iterrows():
            assert len(scenes[(scenes.tag==tag)&(scenes.year==v.year)])==v.scene_count
            assert v.pipeline_sha256=='3cc5e6bc248214e39367b245587f6f19f6d683bf418995285e8b2054e9ba3909'
            assert v.crs=='EPSG:32652' and v['transform']=='30,0,0,0,-30,10000020'
            assert v.buffer_each_side_m==2100 and v.QA_RADSAT_zero==1 and v.QA_PIXEL_bits_0_to_5_clear==1
            assert abs(v.scale_factor-.0000275)<1e-12 and v.offset==-.2
            assert v.start==f'{v.year}-'+('07-01' if tag=='L8_JulAug' else '06-01')
            assert v.end_exclusive==f'{v.year}-'+('09-01' if tag=='L8_JulAug' else '10-01')
    assert not scenes.duplicated(['tag','year','scene_id']).any()
    csv(logs,'export_logs.csv');csv(scenes,'scene_manifest.csv')
    old=pd.read_pickle(PREV/'analysis_inputs.pkl')
    ix=[]
    for branch in ['base','PFR']:
        z=np.load(PREV/'pairs'/f'{branch}.npz');ix.extend([z['target'],z['controls'].ravel()])
    ix=np.unique(np.concatenate(ix));f=old.iloc[ix].copy().reset_index(drop=True)
    del old
    records=[];groups={};duplicates=[]
    for p in paths:
        if p.suffix.lower()!='.tif':continue
        match=PAT.search(p.name);assert match,p
        tag,var,year,row,col=match.groups();year=int(year);key=(tag,var,year,int(row),int(col))
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if key in groups:
            assert sha==groups[key][1],f'Conflicting tile {p}'
            duplicates.append(dict(duplicate=str(p),kept=str(groups[key][0]),sha256=sha));continue
        groups[key]=(p,sha)
    qa=[]
    for tag,ys in [('L8_JulAug',range(2013,2024)),('L7_JunSep',range(2010,2015))]:
      for year in ys:
        seen_by_var={}
        for var in ['NDVI','NOBS']:
            tiles=[(k,v) for k,v in groups.items() if k[:3]==(tag,var,year)]
            assert tiles,(tag,var,year)
            result=np.full(len(f),np.nan,dtype='float32');seen=np.zeros(len(f),np.uint8);foot=[]
            for key,(p,sha) in tiles:
                with rasterio.open(p) as s:
                    assert s.crs.to_epsg()==32652 and s.res==(30,30) and s.count==1
                    assert s.transform.b==0 and s.transform.d==0
                    assert abs(s.transform.c/30-round(s.transform.c/30))<1e-7
                    assert abs((10000020-s.transform.f)/30-round((10000020-s.transform.f)/30))<1e-7
                    assert s.nodata==(-9999 if var=='NDVI' else 65535)
                    a=s.read(1);valid=(a!=s.nodata)&np.isfinite(a)
                    if var=='NDVI':
                        other=groups[(tag,'NOBS',year,key[3],key[4])][0]
                        with rasterio.open(other) as ns:
                            assert ns.shape==s.shape and ns.transform==s.transform
                            nn=ns.read(1)
                            assert np.array_equal(valid,(nn>0)&(nn!=ns.nodata)),'Full-tile NDVI/NOBS mismatch'
                    if var=='NDVI':assert ((a[valid]>=-1)&(a[valid]<=1)).all()
                    else:assert (a[valid]<=logs[(logs.tag==tag)&(logs.year==year)].scene_count.iloc[0]).all()
                    b=s.bounds;sel=(f.x>=b.left)&(f.x<b.right)&(f.y>b.bottom)&(f.y<=b.top)
                    loc=np.flatnonzero(sel);rr=np.floor((b.top-f.y.to_numpy()[loc])/30).astype(int);cc=np.floor((f.x.to_numpy()[loc]-b.left)/30).astype(int)
                    vals=a[rr,cc].astype('float32');vals[vals==s.nodata]=np.nan
                    assert not seen[loc].any(),'Unexpected raster footprint overlap'
                    seen[loc]+=1;result[loc]=vals;foot.append(box(*s.bounds))
                    records.append(dict(tag=tag,variable=var,year=year,path=str(p),sha256=sha,bytes=p.stat().st_size,width=s.width,height=s.height,left=b.left,top=b.top,right=b.right,bottom=b.bottom,valid_pixels=int(valid.sum()),nonfinite_pixels=int((~np.isfinite(a)).sum()),min=float(a[valid].min()) if valid.any() else None,max=float(a[valid].max()) if valid.any() else None))
            assert (seen==1).all(),f'Missing frozen analysis pixels {(tag,var,year)} {(seen==0).sum()}'
            f[f'{tag}_{var}_{year}']=result;seen_by_var[var]=seen
            qa.append(dict(tag=tag,variable=var,year=year,tiles=len(tiles),sampled_pixels=len(f),within_tile_footprints=int(seen.sum()),valid_sampled=int(np.isfinite(result).sum())))
        nd=f[f'{tag}_NDVI_{year}'].to_numpy();ob=f[f'{tag}_NOBS_{year}'].to_numpy()
        assert np.isfinite(ob).all(),'NOBS missing in study samples: missing export footprint?'
        assert np.array_equal(np.isfinite(nd),ob>0),'NDVI / NOBS mask inconsistency'
        if tag=='L8_JulAug':
            assert (ob<=f[f'NOBS_{year}']).all(),'July-Aug counts exceed June-Sept counts'
        print('INGEST',tag,year,'valid',np.isfinite(nd).sum(),flush=True)
    f.to_pickle(OUT/'frozen_pair_pixels_with_new_data.pkl')
    csv(pd.DataFrame(records),'raster_inventory.csv');csv(pd.DataFrame(qa),'raster_group_QA.csv')
    write('ingest_QA.json',dict(archives=archives,duplicates=duplicates,unique_tifs=len(records),logs=len(logs),scenes=len(scenes),sampled_unique_pixels=len(f),all_frozen_pixels_covered=True,mask_consistency=True,grid='EPSG:32652 30m, aligned to original',note='Coverage verified for every frozen target/reference pixel. Whole export rectangle need not be filled: GEE skips empty tiles.'))
    return f

def summarise(a,metrics,label):
    rows=[]
    for route in ['Russia','China']:
      for typ in ['All','Forest','Cropland','Other vegetation']:
        x=a[a.route==route]
        if typ!='All':x=x[x.type==typ]
        if x.empty:continue
        v=x.groupby('unit')[metrics].mean();assert np.isfinite(v).all().all()
        for L in [5,10,20]:
            ids,inv=np.unique(v.index.to_numpy()//L,return_inverse=True);sums=np.zeros((len(ids),len(metrics)));np.add.at(sums,inv,v.to_numpy());counts=np.bincount(inv)
            rng=np.random.default_rng(20261111+L);pick=rng.integers(len(ids),size=(2000,len(ids)));boot=sums[pick].sum(axis=1)/counts[pick].sum(axis=1)[:,None]
            low,high=np.quantile(boot,[.025,.975],axis=0)
            for j,m in enumerate(metrics):rows.append(dict(analysis=label,route=route,type=typ,metric=m,mean=v[m].mean(),ci_low=low[j] if len(ids)>=5 else np.nan,ci_high=high[j] if len(ids)>=5 else np.nan,block_km=L,blocks=len(ids),units=len(v),targets=len(x)))
    return rows

def pairs(f,branch):
    p=pd.read_csv(PREV/'pairs'/f'{branch}.csv.gz');idx=pd.Index(f.pixel_id);t=idx.get_indexer(p.target_id);c=idx.get_indexer(p[[f'control{i}_id' for i in [1,2,3]]].to_numpy().ravel()).reshape(-1,3)
    assert (t>=0).all() and (c>=0).all()
    return t,c

def frame(f,t,c,tag,years):
    nd=f[[f'{tag}_NDVI_{y}' if tag else f'NDVI_{y}' for y in years]].to_numpy()
    ob=f[[f'{tag}_NOBS_{y}' if tag else f'NOBS_{y}' for y in years]].to_numpy()
    a=f.iloc[t][['pixel_id','route','unit','type','lc2013','NDVI_pre','elevation','slope','PFR_pre','PFR_cell_id','longitude','latitude','distance_m','chainage_m']].reset_index(drop=True).copy()
    for j,y in enumerate(years):
        a[f'G_{y}']=nd[t,j]-nd[c,j].mean(axis=1);a[f'T_{y}']=nd[t,j];a[f'R_{y}']=nd[c,j].mean(axis=1)
        a[f'Q_{y}']=(ob[t,j]>=1)&(ob[c,j]>=1).all(axis=1)&np.isfinite(a[f'G_{y}'])
        a[f'N_{y}']=np.minimum(ob[t,j],ob[c,j].min(axis=1))
    if 2013 in years and 2014 in years:
        baseline=a[['G_2013','G_2014']].mean(axis=1,skipna=False)
        for y in years:a[f'D_{y}']=a[f'G_{y}']-baseline
    return a

def analyse(f):
    results=[];ret=[];annualq=[];crossrows=[];crossbal=[];seasonframes={}
    for branch in ['base','PFR']:
        t,c=pairs(f,branch);old=frame(f,t,c,'',list(range(2013,2024)));new=frame(f,t,c,'L8_JulAug',list(range(2013,2024)))
        for country in ['Russia','China']:
            mask=old.route==country
            for year in range(2013,2024):annualq.append(dict(branch=branch,route=country,year=year,matched=int(mask.sum()),JunSep_valid=int(old.loc[mask,f'Q_{year}'].sum()),JulAug_valid=int(new.loc[mask,f'Q_{year}'].sum()),JulAug_minNOBS_median=float(new.loc[mask,f'N_{year}'].median())))
        for end in [2023,2022]:
            years=list(range(2013,end+1));qcols=[f'Q_{y}' for y in years];od=old[qcols].all(axis=1);common=od&new[qcols].all(axis=1)
            for route in ['Russia','China']:
                r=old.route==route;ret.append(dict(branch=branch,end_year=end,route=route,baseline_matches=int(r.sum()),JunSep_complete=int((r&od).sum()),common_complete=int((r&common).sum()),common_units=old[r&common].unit.nunique()))
            for name,a,mask in [('JunSep_original_cohort',old,od),('JunSep_common',old,common),('JulAug_common',new,common)]:
                v=a[mask].copy();metrics=[f'D_{y}' for y in years];v['operation']=v[[f'D_{y}' for y in range(2020,end+1)]].mean(axis=1);metrics+=['operation']
                results+=summarise(v,metrics,f'{branch}_{name}_end{end}')
                if end==2023:seasonframes[(branch,name)]=v;csv(v.groupby(['route','unit']).agg(n=('pixel_id','size'),**{m:(m,'mean') for m in metrics}).reset_index(),f'{branch}_{name}_segments.csv')
            v=new[common].copy()
            for y in years:v[f'D_{y}']=new.loc[common,f'D_{y}']-old.loc[common,f'D_{y}']
            v['operation']=v[[f'D_{y}' for y in range(2020,end+1)]].mean(axis=1)
            results+=summarise(v,[f'D_{y}' for y in years]+['operation'],f'{branch}_JulAug_minus_JunSep_common_end{end}')
            csv(pd.DataFrame(dict(pixel_id=old.pixel_id,JunSep_complete=od,common_complete=common)),f'{branch}_cohort_end{end}.csv.gz')
        pp=pd.read_csv(PREV/f'common_strict_{branch}_pairs.csv.gz')
        a=seasonframes[(branch,'JunSep_common')].set_index('pixel_id',drop=False);b=seasonframes[(branch,'JulAug_common')].set_index('pixel_id',drop=False)
        sel=pp.Russia_target_id.isin(a.index)&pp.China_target_id.isin(a.index);pp=pp[sel]
        for name,fr in [('JunSep',a),('JulAug',b),('Season_difference',b.copy())]:
            if name=='Season_difference':
                for y in range(2013,2024):fr[f'D_{y}']=b[f'D_{y}']-a[f'D_{y}']
            rr,bb=cross_stats(fr.loc[pp.Russia_target_id].reset_index(drop=True),fr.loc[pp.China_target_id].reset_index(drop=True),f'{branch}_{name}_season_common');crossrows+=rr;crossbal+=bb
        csv(pp,f'{branch}_crosscountry_season_common_pairs.csv.gz')
    csv(pd.DataFrame(results),'season_effects.csv');csv(pd.DataFrame(ret),'season_cohort_retention.csv');csv(pd.DataFrame(annualq),'annual_pair_observation_coverage.csv')
    csv(pd.DataFrame(crossrows),'season_crosscountry_effects.csv');csv(pd.DataFrame(crossbal),'season_crosscountry_balance.csv')
    allrows=[];keepcounts=[];sensorrows=[]
    for branch in ['base','PFR']:
        t,c=pairs(f,branch);a=frame(f,t,c,'L7_JunSep',list(range(2010,2015)));o=frame(f,t,c,'',list(range(2013,2024)))
        primary=o[[f'Q_{y}' for y in range(2013,2024)]].all(axis=1)
        for th in [1,3]:
            good=primary&(a[[f'N_{y}' for y in range(2010,2015)]]>=th).all(axis=1)
            good &= a[[f'Q_{y}' for y in range(2010,2015)]].all(axis=1)
            v=a[good].copy();g=v[[f'G_{y}' for y in range(2010,2015)]].to_numpy()
            v['early_slope_2010_2012']=(g[:,2]-g[:,0])/2
            v['slope_2010_2014']=g@np.array([-2,-1,0,1,2])/10
            v['baseline_shift_2013_2014_minus_2010_2012']=g[:,3:].mean(axis=1)-g[:,:3].mean(axis=1)
            metrics=[f'G_{y}' for y in range(2010,2015)]+['early_slope_2010_2012','slope_2010_2014','baseline_shift_2013_2014_minus_2010_2012']
            allrows+=summarise(v,metrics,f'{branch}_L7_fixed_primary_NOBS{th}')
            for route in ['Russia','China']:keepcounts.append(dict(branch=branch,NOBS_threshold=th,route=route,original_complete=int((primary&(o.route==route)).sum()),early_complete=int((good&(o.route==route)).sum()),units=v[v.route==route].unit.nunique()))
            if th==1:
                csv(v.groupby(['route','unit']).agg(n=('pixel_id','size'),**{m:(m,'mean') for m in metrics}).reset_index(),f'{branch}_L7_early_segment_results.csv')
                csv(v,f'{branch}_L7_early_complete_targets.csv.gz')
        overlap=primary&a[['Q_2013','Q_2014']].all(axis=1)
        v=a[overlap].copy()
        for year in [2013,2014]:
            for part in ['T','R','G']:v[f'{part}_L7minusL8_{year}']=a.loc[overlap,f'{part}_{year}']-o.loc[overlap,f'{part}_{year}']
        v['gap_change_difference_2014_minus_2013']=v.G_L7minusL8_2014-v.G_L7minusL8_2013
        metrics=[f'{p}_L7minusL8_{y}' for y in [2013,2014] for p in ['T','R','G']]+['gap_change_difference_2014_minus_2013']
        sensorrows+=summarise(v,metrics,f'{branch}_overlap_common_2013_2014')
    csv(pd.DataFrame(allrows),'L7_early_baseline_effects.csv');csv(pd.DataFrame(keepcounts),'L7_early_cohort_retention.csv');csv(pd.DataFrame(sensorrows),'L7_L8_overlap_diagnostics.csv')
    print('SEASON AND EARLY BASELINE ANALYSES COMPLETE',flush=True)

if __name__=='__main__':
    write('analysis_protocol.json',dict(date='2026-11-11',matching='Frozen original 1:3 base and PFR matches; no rematching using outcomes',season='2013-2023 and 2013-2022 complete common cohorts, paired window contrasts',early='L7-only 2010-2014 gap and slopes; original L8 primary cohort; NOBS1/3',overlap='L7 minus L8 in 2013/2014 on identical pixels; sensor AND sampling difference, not pure sensor calibration',CI='2000 fixed spatial block resamples; 5/10/20km; pointwise',selection_caveat='L7 early histories are selected by 2013/14 L8 matching and later complete cases; not an independent proof of parallel trends'))
    cache=OUT/'frozen_pair_pixels_with_new_data.pkl'
    f=pd.read_pickle(cache) if cache.exists() else ingest()
    analyse(f)
