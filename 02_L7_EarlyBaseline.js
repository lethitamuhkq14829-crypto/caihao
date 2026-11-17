var BATCH = 1;
var BATCHES = [[2010, 2011], [2012, 2013], [2014]];
if (BATCH < 1 || BATCH > BATCHES.length) throw new Error('BATCH 超出范围');
var YEARS = BATCHES[BATCH - 1];
var FOLDER = 'CN_RU_Supp_L7_JunSep_20261109';
var TAG = 'L7_JunSep';
var COLLECTION = 'LANDSAT/LE07/C02/T1_L2';
var START_MONTH = 6;
var END_MONTH_EXCLUSIVE = 10;
var CRS = 'EPSG:32652';
var GRID = [30, 0, 0, 0, -30, 10000020];
var PIPELINE_ASSET = 'REPLACE_WITH_YOUR_ASSET_ID';
if (PIPELINE_ASSET.indexOf('REPLACE_') === 0) throw new Error('Set PIPELINE_ASSET before running.');
var CORRIDOR_ASSET = 'REPLACE_WITH_YOUR_ASSET_ID';
if (CORRIDOR_ASSET.indexOf('REPLACE_') === 0) throw new Error('Set CORRIDOR_ASSET before running.');
var ROI = ee.FeatureCollection(CORRIDOR_ASSET).geometry(1);
var EXPORT_REGION = ee.Geometry.Rectangle([299070.0, 5358510.0, 446400.0, 5718660.0], CRS, false);
var PIPELINE = ee.FeatureCollection(PIPELINE_ASSET).geometry(1);
var logs = [];
var manifests = [];

function prepare(image) {
  var red = image.select('SR_B3').multiply(0.0000275).add(-0.2);
  var nir = image.select('SR_B4').multiply(0.0000275).add(-0.2);
  var clear = image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(image.select('QA_RADSAT').eq(0))
    .and(red.gte(0)).and(nir.gte(0)).and(red.add(nir).gt(0));
  return nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    .updateMask(clear).toFloat().copyProperties(image, ['system:time_start']);
}
function exportImage(image, name, nodata) {
  Export.image.toDrive({
    image: image, description: name, fileNamePrefix: name, folder: FOLDER,
    region: EXPORT_REGION, crs: CRS, crsTransform: GRID,
    maxPixels: 1e10, fileDimensions: 2048, shardSize: 256,
    skipEmptyTiles: true, fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: nodata}
  });
}
YEARS.forEach(function(year) {
  var start = ee.Date.fromYMD(year, START_MONTH, 1);
  var end = ee.Date.fromYMD(year, END_MONTH_EXCLUSIVE, 1);
  var raw = ee.ImageCollection(COLLECTION).filterBounds(ROI)
    .filterDate(start, end).filter(ee.Filter.lt('CLOUD_COVER',70));
  var processed = raw.map(prepare);
  var empty = ee.Image.constant(0).rename('NDVI').toFloat()
    .updateMask(ee.Image.constant(0));
  var safe = processed.merge(ee.ImageCollection.fromImages([empty]));
  var ndvi = safe.median().rename('NDVI').toFloat().clip(ROI);
  var nobs = safe.count().rename('NOBS').unmask(0).toUint16().clip(ROI);
  exportImage(ndvi.unmask(-9999).clip(ROI), 'CN_RU_'+TAG+'_NDVI_'+year+'_30m', -9999);
  exportImage(nobs, 'CN_RU_'+TAG+'_NOBS_'+year+'_30m', 65535);
  logs.push(ee.Feature(null, {
    year: year, tag: TAG, collection: COLLECTION,
    start: start.format('YYYY-MM-dd'), end_exclusive: end.format('YYYY-MM-dd'),
    scene_count: raw.size(), crs: CRS, transform: GRID.join(','),
    composite: 'median_of_valid_scene_NDVI', cloud_cover_lt:70,
    QA_PIXEL_bits_0_to_5_clear:1, QA_RADSAT_zero:1,
    scale_factor:0.0000275, offset:-0.2, reject_negative_reflectance:1,
    buffer_each_side_m:2100, export_footprint_simplification_m:2,
    pipeline_sha256:'3cc5e6bc248214e39367b245587f6f19f6d683bf418995285e8b2054e9ba3909', no_gap_filling:1, batch:BATCH
  }));
  var records = ee.FeatureCollection(raw.toList(raw.size().max(1)).map(function(obj) {
    var im=ee.Image(obj);
    return ee.Feature(null,{year:year,tag:TAG,
      scene_id:im.get('system:index'),date:im.date().format('YYYY-MM-dd'),
      cloud_cover:im.get('CLOUD_COVER'),path:im.get('WRS_PATH'),row:im.get('WRS_ROW')});
  }));
  manifests.push(records);
  print(year+' filtered scenes (NOT per-pixel NOBS)', raw.size());
});
var suffix=TAG+'_B'+BATCH+'_'+YEARS[0]+'_'+YEARS[YEARS.length-1];
Export.table.toDrive({collection:ee.FeatureCollection(logs),description:'CN_RU_log_'+suffix,
  folder:FOLDER,fileNamePrefix:'CN_RU_log_'+suffix,fileFormat:'CSV'});
Export.table.toDrive({collection:ee.FeatureCollection(manifests).flatten(),description:'CN_RU_scenes_'+suffix,
  folder:FOLDER,fileNamePrefix:'CN_RU_scenes_'+suffix,fileFormat:'CSV'});
Map.centerObject(PIPELINE,8);
Map.addLayer(ee.FeatureCollection([ee.Feature(ROI)]).style({color:'5887ae',fillColor:'00000000'}),{},'Export corridor +/-2.1 km');
Map.addLayer(PIPELINE,{color:'red'},'Archived pipeline');
print('本批年份',YEARS);
print('本批应有任务数：2 × 年份数 + 2 CSV',YEARS.length*2+2);
print('一个影像任务会输出多个小 TIFF；全部下载，不要改名或只取其中一块。');
print('点击编辑器 Run 仅创建任务；还需到 Tasks 提交。此脚本不做匹配或计算论文结论。');
