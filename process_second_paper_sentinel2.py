"""Retrieve Sentinel-2 L2A windows and composite daily NDVI on Landsat-aligned grids."""
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import Lock, local
from urllib.parse import urlsplit
import warnings
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import Window
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import shape
from urllib3.util.retry import Retry

from select_second_paper_sentinel2_sites import OUT as SITE_DIR, ROOT


API = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
YEARS = range(2020, 2024)
BANDS = {"B02": 1, "B03": 2, "B04": 3, "B08": 7}
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="22",
           GDAL_HTTP_CONNECTTIMEOUT="8", GDAL_HTTP_MAX_RETRY="1",
           GDAL_HTTP_MULTIRANGE="SERIAL", CPL_VSIL_CURL_USE_HEAD="NO",
           CPL_VSIL_CURL_CHUNK_SIZE="32768", GDAL_NUM_THREADS="1")
NODATA = -9999
THREAD = local()
TOKENS = {}
TOKEN_LOCK = Lock()


@dataclass
class Site:
    name: str
    geometry: object
    geographic: object
    transform: Affine
    width: int
    height: int
    mask: np.ndarray


def read_sites():
    gdf = gpd.read_file(SITE_DIR/"S2_validation_windows.gpkg", layer="windows_utm52")
    ll = gdf.to_crs(4326)
    info = json.loads((SITE_DIR/"site_selection.json").read_text(encoding="utf8"))
    a, _, x0, _, _, y0 = info["landsat_transform"]
    sites = {}
    for i, row in gdf.iterrows():
        xmin, ymin, xmax, ymax = row.geometry.bounds
        left = x0+math.floor((xmin-x0)/a)*a
        top = y0+math.ceil((ymax-y0)/a)*a
        w, h = math.ceil((xmax-left)/a)*3, math.ceil((top-ymin)/a)*3
        affine = Affine(10, 0, left, 0, -10, top)
        inside = geometry_mask([row.geometry], (h,w), affine, invert=True)
        sites[row.site] = Site(row.site,row.geometry,ll.geometry.iloc[i],affine,w,h,inside)
    return sites


def http():
    if not hasattr(THREAD, "http"):
        THREAD.http = requests.Session()
        retries = Retry(total=3, backoff_factor=.6, status_forcelist=[429,500,502,503,504])
        THREAD.http.mount("https://", HTTPAdapter(max_retries=retries))
    return THREAD.http


def get(url, **kwargs):
    r = http().get(url, timeout=(10,30), **kwargs)
    r.raise_for_status()
    return r


def signed(url):
    parsed = urlsplit(url)
    account, container = parsed.netloc.split(".")[0], parsed.path.split("/")[1]
    key = account,container
    with TOKEN_LOCK:
        if key not in TOKENS:
            TOKENS[key] = get(f"{TOKEN}/{account}/{container}").json()["token"].lstrip("?")
    return url+"?"+TOKENS[key]


def catalogue(sites, year):
    file = SITE_DIR/f"PC_catalog_{year}.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf8"))
    bounds = [s.geographic.bounds for s in sites.values()]
    bbox = [min(a[0] for a in bounds),min(a[1] for a in bounds),max(a[2] for a in bounds),max(a[3] for a in bounds)]
    params = dict(collections="sentinel-2-l2a",bbox=",".join(str(round(n,6)) for n in bbox),
                  datetime=f"{year}-07-01T00:00:00Z/{year}-08-31T23:59:59Z",limit=100)
    r = get(API+"/search",params=params)
    items = []
    while True:
        page = r.json()
        items.extend(i for i in page["features"] if i["properties"].get("proj:epsg")==32652
                     and any(shape(i["geometry"]).intersects(s.geographic) for s in sites.values()))
        nxt = next((l["href"] for l in page["links"] if l["rel"]=="next"),None)
        if nxt is None:
            break
        r = get(nxt)
    unique = {}
    for it in sorted(items,key=lambda i:(float(i["properties"]["s2:processing_baseline"]), i["id"])):
        p = it["properties"]
        unique[(p["datetime"],p["s2:mgrs_tile"], it["id"][:3])] = it
    result = sorted(unique.values(),key=lambda i:i["id"])
    file.write_text(json.dumps(result),encoding="utf8")
    print(f"{year}: {len(result)} intersecting native UTM-52N granules",flush=True)
    return result


def calibration(item):
    p = item["properties"]
    asset = item["assets"]["product-metadata"]["href"]
    root = ET.fromstring(get(signed(asset)).content)
    quant = float(root.findtext(".//BOA_QUANTIFICATION_VALUE"))
    offsets = {int(x.attrib["band_id"]): float(x.text) for x in root.findall(".//BOA_ADD_OFFSET")}
    assert quant > 0
    assert offsets or float(p["s2:processing_baseline"])<4
    return quant, {b: offsets.get(n,0) for b,n in BANDS.items()}


def read_nearest(dataset, site):
    """Exact nearest cell-centre sample on the shared EPSG:32652 grid."""
    a = dataset.transform
    assert dataset.crs.to_epsg()==32652 and a.b==a.d==0 and a.a in (10,20) and a.e==-a.a
    x = site.transform.c+10*(np.arange(site.width)+.5)
    y = site.transform.f-10*(np.arange(site.height)+.5)
    cols = np.floor((x-a.c)/a.a).astype(int)
    rows = np.floor((a.f-y)/a.a).astype(int)
    c = (cols>=0)&(cols<dataset.width)
    r = (rows>=0)&(rows<dataset.height)
    result = np.zeros((site.height,site.width),dtype=dataset.dtypes[0])
    if not c.any() or not r.any():
        return result
    c0,c1 = cols[c][[0,-1]]
    r0,r1 = rows[r][[0,-1]]
    values = dataset.read(1,window=Window(c0,r0,c1-c0+1,r1-r0+1))
    result[np.ix_(r,c)] = values[np.ix_(rows[r]-r0,cols[c]-c0)]
    return result


def scene_cache(item, sites, rgb=False):
    item_id = item["id"]
    footprint = shape(item["geometry"])
    matches = [s for s in sites if footprint.intersects(s.geographic)]
    folder = SITE_DIR/("rgb_cache" if rgb else "scene_cache")
    folder.mkdir(exist_ok=True)
    pending = [s for s in matches if not (folder/f"{s.name}_{item_id}.npz").exists()]
    if not pending:
        return
    assets = item["assets"]
    with rasterio.Env(**ENV):
        with rasterio.open(signed(assets["SCL"]["href"])) as scl:
            classes = {s.name:read_nearest(scl,s) for s in pending}
        valid = {s.name: np.isin(classes[s.name], [4,5,6]) & s.mask for s in pending}
        usable = [s for s in pending if valid[s.name].any()]
        values = defaultdict(dict)
        quant, offsets = (calibration(item) if usable else (0,{}))
        for band in (["B04","B03","B02"] if rgb else ["B04","B08"]):
            if not usable:
                break
            with rasterio.open(signed(assets[band]["href"])) as dataset:
                assert dataset.scales[0]==1 and dataset.offsets[0]==0
                for s in usable:
                    dn = read_nearest(dataset,s)
                    sr = (dn.astype("float32")+offsets[band])/quant
                    sr[(dn==0)|~valid[s.name]] = np.nan
                    values[s.name][band] = sr
        for s in pending:
            file = folder/f"{s.name}_{item_id}.npz"
            data = dict(scl=classes[s.name],scale=quant,
                        b04_offset=offsets.get("B04",0), b08_offset=offsets.get("B08",0), **values[s.name])
            with file.with_suffix(".tmp").open("wb") as fp:
                np.savez_compressed(fp,**data)
            file.with_suffix(".tmp").replace(file)


def fetch_all(tasks, workers, rgb=False):
    failures=[]
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(scene_cache,it,sites,rgb):it["id"] for it,sites in tasks}
        for idx, fut in enumerate(as_completed(futures),1):
            try:
                fut.result()
            except Exception as e:
                failures.append(dict(scene=futures[fut],error_type=type(e).__name__))
            if idx%20==0 or idx==len(futures):
                print(f"{'RGB' if rgb else 'Spectral'}: {idx}/{len(futures)} granules, errors {len(failures)}",flush=True)
    return failures


def ndvi(red, nir):
    valid = (red>=0)&(nir>=0)&((red+nir)>0)
    return np.divide(nir-red,nir+red,out=np.full(red.shape,np.nan,dtype="float32"),where=valid)


def write_raster(path, array, affine, descriptions, nodata=NODATA, **tags):
    if array.ndim==2:
        array=array[None]
    h,w=array.shape[1:]
    with rasterio.open(path,"w",driver="GTiff",width=w,height=h,count=len(array),dtype=array.dtype,
                       crs="EPSG:32652",transform=affine,nodata=nodata,compress="deflate",
                       tiled=True,blockxsize=128,blockysize=128) as ds:
        ds.write(array)
        for i,name in enumerate(descriptions,1):
            ds.set_band_description(i,name)
        ds.update_tags(**tags)


def count_and_median(stack):
    a=np.asarray(stack,dtype="float32")
    count=np.isfinite(a).sum(axis=0).astype("float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        med=np.nanmedian(a,axis=0)
    return med,count


def priority(it):
    return (float(it["properties"].get("eo:cloud_cover") or 0),it["id"])


def composite(site, items, year):
    directory=SITE_DIR/"products"/site.name/str(year)
    directory.mkdir(parents=True,exist_ok=True)
    by_date=defaultdict(list)
    shape_10=(site.height,site.width)
    per_scene=[]
    cache=SITE_DIR/"scene_cache"
    for it in sorted(items,key=priority):
        if not shape(it["geometry"]).intersects(site.geographic):
            continue
        file=cache/f"{site.name}_{it['id']}.npz"
        date=it["properties"]["datetime"][:10]
        status="missing" if not file.exists() else "read"
        props=it["properties"]
        rec=dict(site=site.name,year=year,date=date,scene=it["id"],status=status,
                 baseline=props["s2:processing_baseline"],tile=props["s2:mgrs_tile"],
                 tile_cloud_cover=props.get("eo:cloud_cover"),valid_pixels=0,quantification=np.nan,
                 red_dn_offset=np.nan,nir_dn_offset=np.nan)
        if file.exists():
            with np.load(file) as z:
                if "B04" in z:
                    red,nir=z["B04"],z["B08"]
                    keep=np.isfinite(ndvi(red,nir))
                    rec.update(valid_pixels=int(keep.sum()),quantification=float(z["scale"]),
                               red_dn_offset=float(z["b04_offset"]),nir_dn_offset=float(z["b08_offset"]))
                    if keep.any():
                        by_date[date].append((it,np.where(keep,red,np.nan),np.where(keep,nir,np.nan)))
        per_scene.append(rec)
    a10, a30, rgb_candidates = [], [], []
    for date, group in sorted(by_date.items()):
        red=np.full(shape_10,np.nan,dtype="float32")
        nir=red.copy()
        for it, r, n in group:
            take=np.isnan(red)&np.isfinite(r)
            red[take],nir[take]=r[take],n[take]
            rgb_candidates.append((int(np.isfinite(r).sum()),it))
        a10.append(ndvi(red,nir))
        h,w=site.height//3,site.width//3
        # Full 3x3 support keeps partial cloud gaps out of the 30 m comparison.
        def aggregate(band):
            return band.reshape(h,3,w,3).mean(axis=(1,3))
        a30.append(ndvi(aggregate(red),aggregate(nir)))
    if not a10:
        raise RuntimeError(f"No valid data at {site.name} in {year}")
    best_it=max(rgb_candidates,key=lambda x:(x[0],-priority(x[1])[0],x[1]["id"]))[1]
    row=dict(site=site.name,year=year,scenes=len(per_scene),missing=sum(i["status"]=="missing" for i in per_scene),
             valid_dates=len(by_date),rgb_scene=best_it["id"],rgb_date=best_it["properties"]["datetime"][:10])
    for res, stack, geom_mask in [(10,a10,site.mask),(30,a30,site.mask.reshape(site.height//3,3,site.width//3,3).all(axis=(1,3)))]:
        m,n=count_and_median(stack)
        assert np.nanmin(m)>=-1 and np.nanmax(m)<=1 and n.max()<=len(by_date)
        out=np.array([np.where(geom_mask&np.isfinite(m),m,NODATA),np.where(geom_mask,n,NODATA)],dtype="float32")
        write_raster(directory/f"{site.name}_{year}_NDVI_NOBS_{res}m.tif",out,site.transform*Affine.scale(res/10),
                     ("NDVI","NOBS_DAY"),source="Copernicus Sentinel-2 L2A via Microsoft Planetary Computer",
                     window="July-August",scl_valid="4,5,6",composite="median_of_daily_NDVI")
        row[f"px_{res}m"]=int(geom_mask.sum())
        row[f"pct_ge3_{res}m"]=100*float((n[geom_mask]>=3).mean())
        row[f"median_nobs_{res}m"]=float(np.median(n[geom_mask]))
    return row,per_scene,best_it


def export_rgb(site, year, it):
    p=SITE_DIR/"products"/site.name/str(year)
    file=SITE_DIR/"rgb_cache"/f"{site.name}_{it['id']}.npz"
    with np.load(file) as z:
        bands=np.asarray([z[k] for k in ["B04","B03","B02"]],dtype="float32")
    assert np.isfinite(bands).any()
    write_raster(p/f"{site.name}_{year}_RGB_10m.tif",np.where(np.isfinite(bands),bands,NODATA),
                 site.transform,("red_reflectance","green_reflectance","blue_reflectance"),
                 scene=it["id"],date=it["properties"]["datetime"][:10])


def main():
    ap=ArgumentParser()
    ap.add_argument("--workers",type=int,default=18)
    ap.add_argument("--year",type=int,choices=list(YEARS),action="append")
    args=ap.parse_args()
    years=args.year or list(YEARS)
    sites=read_sites()
    with ThreadPoolExecutor(4) as pool:
        items=dict(zip(years,pool.map(lambda y:catalogue(sites,y),years)))
    tasks=[(it,list(sites.values())) for group in items.values() for it in group]
    errors=fetch_all(tasks,args.workers)
    if errors:
        ids={r["scene"] for r in errors}
        errors=fetch_all([job for job in tasks if job[0]["id"] in ids],max(4,args.workers//2))
    (SITE_DIR/"spectral_errors.json").write_text(json.dumps(errors,indent=2),encoding="utf8")
    data, scenes, rgb_jobs = [], [], []
    for yr, group in items.items():
        for s in sites.values():
            row,rows,it=composite(s,group,yr)
            data.append(row)
            scenes.extend(rows)
            rgb_jobs.append((s,yr,it))
    pd.DataFrame(data).to_csv(SITE_DIR/"S2_site_year_quality.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(scenes).to_csv(SITE_DIR/"S2_scene_quality.csv",index=False,encoding="utf-8-sig")
    jobs=defaultdict(list)
    rgb_items={}
    for s,yr,it in rgb_jobs:
        rgb_items[it["id"]]=it
        jobs[it["id"]].append(s)
    errors_rgb=fetch_all([(rgb_items[i],owners) for i,owners in jobs.items()],args.workers,rgb=True)
    if errors_rgb:
        errors_rgb=fetch_all([(rgb_items[r["scene"]],jobs[r["scene"]]) for r in errors_rgb],max(4,args.workers//2),rgb=True)
    (SITE_DIR/"rgb_errors.json").write_text(json.dumps(errors_rgb,indent=2),encoding="utf8")
    for s,yr,it in rgb_jobs:
        export_rgb(s,yr,it)
    print(f"Finished {len(data)} site-years; {len(errors)} remaining spectral download errors.",flush=True)


if __name__=="__main__":
    main()
