"""Validate the operation-period spatial gradient on identical Landsat footprints."""
from pathlib import Path
import json

from affine import Affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
import shapely

from process_second_paper_sentinel2 import SITE_DIR, ROOT, YEARS, read_sites
from analyze_crossborder_gee_ndvi_20261106 import get_geometry


DEST = ROOT / "output/tables/SecondPaper_Sentinel2_20261114"
SEASON = ROOT / "output/tables/SecondPaper_SeasonBaseline_20261111/raster_inventory.csv"
ARCHIVE = ROOT / "output/tables/SecondPaper_Complete_20261109"
LEVELS = [0, 30, 60, 90, 150, 300, 500]
GROUPS = {"Forest": [51,52,61,62,71,72,81,82,91,92], "Cropland": [10,11,12,20]}
SENSORS = ["S2_10m", "S2_30m", "L8_30m"]
MIN_PIXELS = 10


def centre_grid(a, shape):
    rows, cols = np.indices(shape)
    return a * (cols+.5, rows+.5)


def read_covered_tile(path, a, dims, output):
    height, width = dims
    with rasterio.open(path) as ds:
        assert ds.crs.to_epsg()==32652 and ds.res==(30,30)
        offx, offy = (~ds.transform)*(a.c,a.f)
        col, row = int(round(offx)),int(round(offy))
        assert abs(offx-col)<1e-7 and abs(offy-row)<1e-7
        c0,r0=max(0,col),max(0,row)
        c1,r1=min(ds.width,col+width),min(ds.height,row+height)
        if c1<=c0 or r1<=r0:
            return
        ref = output[r0-row:r1-row,c0-col:c1-col]
        assert np.isnan(ref).all()
        ref[:] = ds.read(1,window=Window(c0,r0,c1-c0,r1-r0),masked=True).astype("float32").filled(np.nan)


def tile_product(table, year, band, affine, dims):
    data=np.full(dims,np.nan,dtype="float32")
    for p in table[(table.year==year)&(table.variable==band)].path:
        read_covered_tile(p,affine,dims,data)
    return data


def repeated(array):
    return array.repeat(3,axis=0).repeat(3,axis=1)


def raster_qc(path, site, scale):
    with rasterio.open(path) as ds:
        assert ds.count==2 and ds.transform==site.transform*Affine.scale(scale/10)
        assert ds.crs.to_epsg()==32652 and ds.descriptions==("NDVI","NOBS_DAY")
        v,n=ds.read(masked=True).filled(np.nan)
    assert np.array_equal(np.isfinite(v), np.isfinite(n)&(n>0))
    assert np.nanmin(v)>=-1 and np.nanmax(v)<=1 and np.all(n[np.isfinite(n)]%1==0)
    return v,n


def matched_gap(pairs, values, support):
    obs=[]
    for i in range(4):
        rows=pairs[f"row_{i}"].to_numpy()
        cols=pairs[f"col_{i}"].to_numpy()
        obs.append(np.where(support[rows,cols],values[rows,cols],np.nan))
    obs=np.asarray(obs)
    ok=np.isfinite(obs).all(axis=0)
    gap=obs[0,ok]-obs[1:,ok].mean(axis=0)
    return float(gap.mean()) if ok.any() else np.nan, int(ok.sum())


def in_window_pairs(pixels, pairs, window, dims, classes):
    chosen=pixels.loc[pixels.distance_m.le(50)&pixels.lc2013.isin(classes)]
    xy=np.array([(~window)*(x,y) for x,y in zip(chosen.x,chosen.y)])
    if not len(xy):
        return pd.DataFrame()
    cols,rows=np.floor(xy).astype(int).T
    within=(rows>=0)&(rows<dims[0])&(cols>=0)&(cols<dims[1])
    selected=pairs.loc[pairs.index.intersection(chosen.index[within])]
    out=pd.DataFrame(index=selected.index)
    for i,ids in enumerate([selected.index]+[selected[f"control{j}_id"] for j in [1,2,3]]):
        coords=pixels.loc[ids,["x","y"]]
        rr,cc=rasterio.transform.rowcol(window,coords.x,coords.y)
        out[f"row_{i}"]=rr
        out[f"col_{i}"]=cc
    for i in range(4):
        out=out[out[f"row_{i}"].between(0,dims[0]-1)&out[f"col_{i}"].between(0,dims[1]-1)]
    return out


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    sites=read_sites()
    properties=gpd.read_file(SITE_DIR/"S2_validation_windows.gpkg",layer="windows_utm52").set_index("site")
    _,(russia,_,china)=get_geometry()
    lines={"Russia":russia,"China":china}
    layers=pd.read_csv(SEASON).query("tag == 'L8_JulAug' and year >= 2020")
    with open(ROOT/"output/reports/SecondPaper_Complete_20261109/input_manifest.json",encoding="utf8") as fp:
        lc=Path(json.load(fp)["files"]["lc2013"]["path"])
    metadata=pd.read_csv(SITE_DIR/"S2_site_year_quality.csv")
    assert len(metadata)==48 and not metadata.duplicated(["site","year"]).any()
    pixels=pd.read_pickle(ARCHIVE/"analysis_inputs.pkl")[["pixel_id","x","y","distance_m","lc2013","route"]].set_index("pixel_id")
    pairs=pd.read_csv(ARCHIVE/"pairs/base.csv.gz").set_index("target_id")
    data, gaps, coverage=[],[],[]
    for name,s in sites.items():
        country=properties.loc[name,"route"]
        vegetation=properties.loc[name,"vegetation"]
        line=lines[country]
        a10=s.transform
        a30=a10*Affine.scale(3)
        dims=(s.height//3,s.width//3)
        xyz10=centre_grid(a10,(s.height,s.width))
        xyz30=centre_grid(a30,dims)
        d10=shapely.distance(line,shapely.points(*xyz10))
        d30=shapely.distance(line,shapely.points(*xyz30))
        lc_array=np.full(dims,np.nan,dtype="float32")
        read_covered_tile(lc,a30,dims,lc_array)
        group_mask=np.isin(lc_array,GROUPS[vegetation])
        local=in_window_pairs(pixels[pixels.route.eq(country)],pairs,a30,dims,GROUPS[vegetation])
        for year in YEARS:
            base=SITE_DIR/"products"/name/str(year)
            v10,n10=raster_qc(base/f"{name}_{year}_NDVI_NOBS_10m.tif",s,10)
            v30,n30=raster_qc(base/f"{name}_{year}_NDVI_NOBS_30m.tif",s,30)
            rgb=base/f"{name}_{year}_RGB_10m.tif"
            with rasterio.open(rgb) as ds:
                assert ds.count==3 and ds.crs.to_epsg()==32652 and ds.transform==a10
                assert np.isfinite(ds.read(masked=True).filled(np.nan)).any()
            vl=tile_product(layers,year,"NDVI",a30,dims)
            nl=tile_product(layers,year,"NOBS",a30,dims)
            s2=(n30>=3)&(n10.reshape(dims[0],3,dims[1],3)>=3).all(axis=(1,3))
            common=s2&(nl>=1)&np.isfinite(vl)&group_mask
            coverage.append(dict(site=name,year=year,country=country,vegetation=vegetation,
                                 focal_landcover_30m=int(group_mask.sum()),s2_ge3_30m=int((s2&group_mask).sum()),
                                 shared_30m=int(common.sum()),available_match_sets=len(local)))
            for sensor,values in [("S2_30m",v30),("L8_30m",vl)]:
                val,n=matched_gap(local,values,common) if len(local) else (np.nan,0)
                gaps.append(dict(site=name,year=year,country=country,vegetation=vegetation,sensor=sensor,
                                 matched_sets=n,gap=val))
            for label,sensors,mask in [
                ("Sentinel overlap",["S2_10m","S2_30m"],s2&group_mask),
                ("Cross-sensor overlap",SENSORS,common),
            ]:
                series={"S2_10m":(v10,d10,repeated(mask)), "S2_30m":(v30,d30,mask), "L8_30m":(vl,d30,mask)}
                for sensor in sensors:
                    values,dist,support=series[sensor]
                    support=support&np.isfinite(values)
                    ref=support&(dist>=500)&(dist<=2000)
                    if ref.sum()<MIN_PIXELS:
                        continue
                    bg=float(values[ref].mean())
                    for lo,hi in zip(LEVELS[:-1],LEVELS[1:]):
                        selected=support&((dist>=lo) if lo==0 else (dist>lo))&(dist<=hi)
                        if selected.sum()<MIN_PIXELS:
                            continue
                        data.append(dict(site=name,year=year,country=country,vegetation=vegetation,
                                         overlap=label,sensor=sensor,distance_lo=lo,distance_hi=hi,
                                         n_pixels=int(selected.sum()),n_reference=int(ref.sum()),
                                         ndvi=float(values[selected].mean()),ref_ndvi=bg,
                                         gap=float(values[selected].mean())-bg))
    detail=pd.DataFrame(data)
    matched=pd.DataFrame(gaps)
    for label,k in [("Sentinel overlap",2),("Cross-sensor overlap",3)]:
        subset=detail[detail.overlap.eq(label)]
        counts=subset.groupby(["site","year"]).size()
        good=counts[counts==k*6].index
        ix=detail.set_index(["site","year"]).index
        detail.loc[detail.overlap.eq(label),"complete_profile"]=ix[detail.overlap.eq(label)].isin(good)
    detail["complete_profile"]=detail.complete_profile.astype(bool)
    site_profile=detail[detail.complete_profile].groupby(
        ["site","country","vegetation","overlap","sensor","distance_lo","distance_hi"],as_index=False
    ).agg(gap=("gap","mean"),years=("year","nunique"))
    profile=site_profile.groupby(["country","overlap","sensor","distance_lo","distance_hi"],as_index=False).agg(
        mean_gap=("gap","mean"),sites=("site","nunique"),min_years=("years","min"),max_years=("years","max"))
    wide=matched.pivot(index=["site","country","vegetation","year","matched_sets"],columns="sensor",values="gap").reset_index()
    wide["S2_minus_L8"]=wide.S2_30m-wide.L8_30m
    for name,frame in [("S2_distance_profiles_by_year",detail),("S2_distance_profiles_by_site",site_profile),
                       ("S2_distance_profiles_summary",profile),("S2_matched_gap_same_pixels",wide),
                       ("S2_Landsat_overlap",pd.DataFrame(coverage))]:
        frame.to_csv(DEST/f"{name}.csv",index=False,encoding="utf-8-sig",float_format="%.9g")
    print(profile[profile.distance_hi.eq(30)].to_string(index=False))
    print(wide[wide.matched_sets.ge(10)].groupby("country")[["S2_30m","L8_30m","S2_minus_L8"]].agg(["size","mean"]))


if __name__=="__main__":
    main()
