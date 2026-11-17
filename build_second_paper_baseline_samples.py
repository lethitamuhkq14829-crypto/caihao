from pathlib import Path
from contextlib import ExitStack
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
import shapely
from shapely.geometry import box
from pyproj import Transformer
from netCDF4 import Dataset

DEFAULT_PROJECT=Path(__file__).resolve().parent
YEARS=list(range(2013,2024))
VEGETATION={10,11,12,20,51,52,61,62,71,72,81,82,91,92,120,121,122,130,140,181,182}

def nearest_index(axis,values):
    i=np.searchsorted(axis,values).clip(1,len(axis)-1)
    return np.where(np.abs(values-axis[i-1])<=np.abs(values-axis[i]),i-1,i)

class NativePFR:
    def __init__(self,root,area_lonlat):
        self.tiles=[]
        for year in [2013,2014]:
            p=next((root/'data/permafrost_extent').glob('*PP-'+str(year)+'-*'))
            with Dataset(p) as ds:
                lat=np.asarray(ds.variables['lat'][:]);lon=np.asarray(ds.variables['lon'][:])
                assert np.all(np.diff(lat)>0) and np.all(np.diff(lon)>0)
                assert ds.variables['PFR'].units=='percent'
                a,b,c,d=area_lonlat
                r0=max(0,int(np.searchsorted(lat,b)) -2);r1=min(len(lat),int(np.searchsorted(lat,d))+2)
                c0=max(0,int(np.searchsorted(lon,a))-2);c1=min(len(lon),int(np.searchsorted(lon,c))+2)
                values=np.ma.filled(ds.variables['PFR'][0,r0:r1,c0:c1].astype('float32'),np.nan)
            if self.tiles:
                assert np.array_equal(lat,self.lat) and np.array_equal(lon,self.lon)
            self.lat,self.lon=lat,lon
            self.tiles.append((p,values,r0,c0))
    def sample(self,lon,lat):
        rr=nearest_index(self.lat,lat);cc=nearest_index(self.lon,lon);vals=[]
        for _,v,r0,c0 in self.tiles:
            x=v[rr-r0,cc-c0].copy();x[(x<0)|(x>100)]=np.nan;vals.append(x)
        valid=np.isfinite(vals[0])&np.isfinite(vals[1])
        mean=np.where(valid,(vals[0]+vals[1])/2,np.nan)
        return mean,rr.astype('int64')*len(self.lon)+cc,valid

def slope_horn(dem,win):
    pad=Window(win.col_off-1,win.row_off-1,win.width+2,win.height+2)
    a=dem.read(1,window=pad,boundless=True,fill_value=dem.nodata).astype('float64')
    a[(a==dem.nodata)|~np.isfinite(a)]=np.nan
    z1,z2,z3=a[:-2,:-2],a[:-2,1:-1],a[:-2,2:]
    z4,z5,z6=a[1:-1,:-2],a[1:-1,1:-1],a[1:-1,2:]
    z7,z8,z9=a[2:,:-2],a[2:,1:-1],a[2:,2:]
    dx=((z3+2*z6+z9)-(z1+2*z4+z7))/(8*dem.res[0])
    dy=((z7+2*z8+z9)-(z1+2*z2+z3))/(8*dem.res[1])
    result=np.degrees(np.arctan(np.hypot(dx,dy)))
    result[~np.isfinite(z1+z2+z3+z4+z5+z6+z7+z8+z9)]=np.nan
    return np.where(np.isfinite(result),result,-9999).astype('float32')

def country_geometries(root,bounds):
    area=box(bounds[0]-.5,bounds[1]-.5,bounds[2]+.5,bounds[3]+.5)
    china=root/'output/deliverables/User_Boundaries_SCS_20261107/01_Original/China/China_Provinces_Original.shp'
    world=root/'output/deliverables/China_Russia_GlobPermafrost_Processed/01_Original_Data/Boundaries_technical_reference/admin0/ne_10m_admin_0_countries.shp'
    cn=gpd.read_file(china).to_crs(4326)
    ru=gpd.read_file(world);ru=ru[ru.ADM0_A3=='RUS'].to_crs(4326)
    geometries=[]
    for g in [cn,ru]:
        pieces=g.geometry.intersection(area);pieces=pieces[~pieces.is_empty]
        union=shapely.union_all(shapely.make_valid(pieces.array))
        geometries.append(gpd.GeoSeries([union],crs=4326).to_crs(32652).iloc[0])
    return geometries,[str(china),str(world)]

def run(args):
    root=Path(args.project).resolve();source=Path(args.source).resolve()
    sys.path.insert(0,str(root/'scripts'))
    import analyze_crossborder_gee_ndvi_20261106 as old
    old.ROOT=root
    digest,_=old.verify_logs(source/'第一步')
    full,parts=old.get_geometry();units=old.get_segment_units(parts)
    buffers=[full.buffer(d,quad_segs=64)for d in old.EDGES_M]
    part_buffers=[p.buffer(10000.1,quad_segs=64)for p in parts]
    lonlat=gpd.GeoSeries([full.buffer(10000)],crs=32652).to_crs(4326).total_bounds
    pfr=NativePFR(root,lonlat);countries,country_sources=country_geometries(root,lonlat)
    to_ll=Transformer.from_crs(32652,4326,always_xy=True)
    files={f'NDVI_{y}':source/f'第一步/CN_RU_NDVI_{y}_30m.tif'for y in YEARS}
    files.update({f'NOBS_{y}':source/f'第一步/CN_RU_NOBS_{y}_30m.tif'for y in YEARS})
    files.update(lc2013=source/'第二步/CN_RU_GLC_FCS30D_2013_30m.tif',lc2014=source/'第二步/CN_RU_GLC_FCS30D_2014_30m.tif',elevation=source/'第二步/CN_RU_SRTM_DEM_30m.tif')
    with ExitStack() as stack:
        ds={k:stack.enter_context(rasterio.open(p))for k,p in files.items()}
        base=ds['NDVI_2013']
        for name,s in ds.items():
            assert (s.crs,s.shape,s.transform,s.count)==(base.crs,base.shape,base.transform,1),name
        assert base.crs.to_epsg()==32652 and base.res==(30,30)
        if args.check:
            line=gpd.GeoSeries([parts[0]],crs=32652).to_crs(4326).iloc[0]
            pp=np.array(line.coords)[::20];v,ids,valid=pfr.sample(pp[:,0],pp[:,1])
            assert np.all((v[valid]>=0)&(v[valid]<=100))
            w=Window(2000,3000,64,64);s=slope_horn(ds['elevation'],w)
            assert s.shape==(64,64)
            print(json.dumps({'check':'PASS','rasters':len(ds),'grid':str(base.crs),'rows':base.height,'columns':base.width,'pipeline_km':full.length/1000,'PFR_test_points':len(v),'PFR_valid_test_points':int(valid.sum()),'matching_has_not_run':True},ensure_ascii=False,indent=2))
            return
        output=Path(args.output) if args.output else source/'第三步_建前样本库'
        if output.exists() and any(output.iterdir()):raise RuntimeError('输出文件夹非空。请换一个 --output 路径，以免覆盖已有结果。')
        output.mkdir(exist_ok=True,parents=True);shards=output/'samples';shards.mkdir()
        profile=base.profile.copy();profile.update(dtype='float32',count=1,nodata=-9999,compress='DEFLATE')
        slope=stack.enter_context(rasterio.open(output/'CN_RU_Slope_Horn_degrees_30m.tif','w',**profile))
        inventory=[];totals=[]
        for start in range(0,base.height,256):
            win=Window(0,start,base.width,min(256,base.height-start));aff=base.window_transform(win)
            sl=slope_horn(ds['elevation'],win);slope.write(sl,1,window=win)
            domain=ds['NOBS_2013'].read(1,window=win)!=65535
            area=rasterize([(buffers[4],1)],out_shape=domain.shape,transform=aff,dtype='uint8')>0
            domain &= area
            if not domain.any():continue
            route,band,points,inside=old.raster_regions(domain.shape,aff,domain,buffers,parts,part_buffers,full)
            rr,cc=np.nonzero(domain);rr,cc=rr[inside],cc[inside]
            take=(band==0)|(band==4)
            rr,cc,route,band,points=rr[take],cc[take],route[take],band[take],points[take]
            if not len(rr):continue
            x,y=aff*(cc+.5,rr+.5);lon,lat=to_ll.transform(x,y)
            pf,pf_id,pf_valid=pfr.sample(lon,lat)
            in_cn=shapely.contains_xy(countries[0],x,y);in_ru=shapely.contains_xy(countries[1],x,y)
            country=np.where(in_cn&~in_ru,'China',np.where(in_ru&~in_cn,'Russia','Ambiguous'))
            route_name=np.array(old.SIDES)[route]
            f=pd.DataFrame(dict(pixel_id=(rr.astype('int64')+start)*base.width+cc,row=rr+start,col=cc,x=x,y=y,longitude=lon,latitude=lat,route=route_name,country=country,country_agrees=(country==route_name),belt=np.where(band==0,'near_0_50m','reference_500_2000m'),segment_id=old.segment_ids(points,route,units),slope=sl[rr,cc],PFR_pre=pf,PFR_cell_id=pf_id,PFR_pre_valid=pf_valid))
            f['border_distance_m']=shapely.distance(points,countries[0].boundary)
            f['border_within_3km']=f.border_distance_m<=3000
            for key,s in ds.items():
                v=s.read(1,window=win)[rr,cc]
                if key.startswith('NDVI_'):v=np.where(v==s.nodata,np.nan,v)
                f[key]=v
            obs=f[[f'NOBS_{y}'for y in YEARS]].to_numpy()
            ndvi=f[[f'NDVI_{y}'for y in YEARS]].to_numpy()
            assert np.array_equal(np.isfinite(ndvi),obs>=1)
            f['min_NOBS_2013_2023']=obs.min(axis=1)
            f['min_NOBS_2013_2022']=obs[:,:-1].min(axis=1)
            pre=ndvi[:,:2];f['NDVI_pre']=np.where(np.isfinite(pre).all(axis=1),pre.mean(axis=1),np.nan)
            f['vegetation_2013']=f.lc2013.isin(VEGETATION)
            f['landcover_baseline_same']=f.lc2013==f.lc2014
            f['eligible_primary']=(f.vegetation_2013&f.country_agrees&f.PFR_pre_valid&(f.slope>=0)&np.isfinite(f.NDVI_pre)&(f.min_NOBS_2013_2023>=1)&(f.route!='Connector'))
            path=shards/f'samples_row_{start:05d}.csv.gz'
            f.to_csv(path,index=False,compression='gzip',float_format='%.8g')
            inventory.append(dict(file=path.name,rows=len(f),eligible_primary=int(f.eligible_primary.sum())))
            totals.append(f.groupby(['route','belt']).agg(pixels=('pixel_id','size'),eligible_primary=('eligible_primary','sum'),border_within_3km=('border_within_3km','sum')).reset_index())
            print('rows processed',start,'sample rows',len(f),'eligible',int(f.eligible_primary.sum()),flush=True)
        summary=pd.concat(totals).groupby(['route','belt'],as_index=False).sum()
        summary.to_csv(output/'sample_counts.csv',index=False,encoding='utf-8-sig')
        meta=dict(task='Build baseline matching inputs only; matching has not run',pipeline_km=full.length/1000,pipeline_geojson_sha256=digest,years=YEARS,near_m=[0,50],reference_m=[500,2000],PFR_years=[2013,2014],PFR_method='mean only where both original source cells are valid; no raster interpolation',country_sources=country_sources,country_ambiguity='overlap/gap is flagged and not eligible',border_sensitivity_flag_m=3000,eligible_primary='baseline vegetation + valid paired PFR + terrain + both baseline NDVI + NOBS>=1 every year + actual country agrees with route; connector excluded',slope='Horn 3x3 in projected meters; all 9 DEM neighbors required',outputs=inventory,input_files={k:str(p)for k,p in files.items()})
        (output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
        print('DONE',output,'sample rows',sum(x['rows']for x in inventory),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Task 3.1: build real baseline covariates and annual NDVI/NOBS sample shards.\n\nRun locally, not in the Earth Engine JavaScript editor. Does not perform matching.\nAll raw inputs are read-only. Each CSV shard can later be matched by pixel_id.\n')
    p.add_argument('--project',default=str(DEFAULT_PROJECT));p.add_argument('--source',default=str(DEFAULT_PROJECT / 'GEE_downloads'));p.add_argument('--output');p.add_argument('--check',action='store_true',help='validate inputs and sample PFR/slope; do not build the full library')
    run(p.parse_args())
