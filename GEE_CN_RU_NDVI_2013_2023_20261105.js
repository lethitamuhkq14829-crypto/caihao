var RUN_ALL_YEARS = false;
var TEST_YEARS = [2013, 2023];
var START_YEAR = 2013;
var END_YEAR = 2023;
var START_MONTH = 6;
var END_MONTH = 9;
var BUFFER_M = 10000;
var SCENE_CLOUD_MAX = 70;
var MIN_VALID_OBS = 1;
var LOW_OBS_WARNING = 3;
var CRS = 'EPSG:32652';
var GRID = [30, 0, 0, 0, -30, 10000020];
var TAG = RUN_ALL_YEARS ? 'ALL_2013_2023' : 'TEST';
var DRIVE_FOLDER = 'CN_RU_NDVI_v1_20261105_' + TAG;
var NDVI_NODATA = -9999;
var COUNT_NODATA = 65535;

var PIPELINE_ASSET = 'REPLACE_WITH_YOUR_ASSET_ID';
if (PIPELINE_ASSET.indexOf('REPLACE_') === 0) throw new Error('Set PIPELINE_ASSET before running.');
var projection = ee.Projection(CRS);
var exportProjection = ee.Projection(CRS, GRID);
var pipeline = ee.FeatureCollection(PIPELINE_ASSET).geometry(1);
var corridor = pipeline.buffer(BUFFER_M, 1, projection);
var exportRegion = corridor.bounds(1, projection);
var years = TEST_YEARS.slice();
if (RUN_ALL_YEARS) {
  years = [];
  for (var y = START_YEAR; y <= END_YEAR; y++) years.push(y);
}
if (START_MONTH < 1 || END_MONTH > 12 || START_MONTH > END_MONTH) {
  throw new Error('月份设置错误');
}
if (MIN_VALID_OBS < 1 || BUFFER_M <= 0) throw new Error('观测数或缓冲距离设置错误');
years.forEach(function(y) {
  if (y < 2013 || y > END_YEAR || Math.floor(y) !== y) throw new Error('请检查年份');
});

Map.centerObject(pipeline, 7);
Map.addLayer(ee.FeatureCollection([ee.Feature(corridor)]).style({
  color: '777777', fillColor: '00000000', width: 1
}), {}, '10 km data corridor', false);
print('当前模式 / 年份', TAG, years);
print('完整研究管线长度 km（应约428.16）', pipeline.length(1, projection).divide(1000));
print('数据目录', DRIVE_FOLDER);
print('固定输出投影 / 网格', CRS, GRID);
print('注意：Count表示有效影像观测次数，不保证是互不重复的日期数。');

function sceneNDVI(image) {
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(63).eq(0);
  var radMask = image.select('QA_RADSAT').eq(0);
  var red = image.select('SR_B4').multiply(0.0000275).add(-0.2);
  var nir = image.select('SR_B5').multiply(0.0000275).add(-0.2);
  var denominator = nir.add(red);
  var valid = qaMask.and(radMask).and(red.gte(0)).and(nir.gte(0))
    .and(denominator.gt(0));
  var ndvi = nir.subtract(red).divide(denominator)
    .updateMask(valid).rename('NDVI').toFloat();
  return ndvi.copyProperties(image, ['system:time_start', 'system:index']);
}

function exportRaster(image, name, noData) {
  Export.image.toDrive({
    image: image.unmask({value: noData, sameFootprint: false}),
    description: name,
    fileNamePrefix: name,
    folder: DRIVE_FOLDER,
    region: exportRegion,
    crs: CRS,
    crsTransform: GRID,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: noData}
  });
}

var manifest = ee.FeatureCollection([]);
var yearSummaries = [];
var ndviVis = {min: -0.1, max: 0.85,
  palette: ['8c510a', 'dfc27d', 'f6e8c3', 'c7eae5', '5ab4ac', '01665e']};

years.forEach(function(year) {
  var begin = ee.Date.fromYMD(year, START_MONTH, 1);
  var end = ee.Date.fromYMD(year, END_MONTH, 1).advance(1, 'month');
  var scenes = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(corridor).filterDate(begin, end)
    .filter(ee.Filter.lt('CLOUD_COVER', SCENE_CLOUD_MAX));
  var empty = ee.Image.constant(0).rename('NDVI').toFloat()
    .updateMask(ee.Image.constant(0)).setDefaultProjection(exportProjection);
  var observations = scenes.map(sceneNDVI)
    .merge(ee.ImageCollection.fromImages([empty]));
  var nobs = observations.select('NDVI').count()
    .setDefaultProjection(exportProjection).unmask(0)
    .clip(corridor).rename('NOBS').toUint16();
  var annual = observations.select('NDVI').median()
    .setDefaultProjection(exportProjection)
    .updateMask(nobs.gte(MIN_VALID_OBS)).clip(corridor)
    .rename('NDVI').toFloat()
    .set({year: year, season_start: begin.format('YYYY-MM-dd'),
      season_end_exclusive: end.format('YYYY-MM-dd'),
      'system:time_start': begin.millis()});

  exportRaster(annual, 'CN_RU_NDVI_' + year + '_30m', NDVI_NODATA);
  exportRaster(nobs, 'CN_RU_NOBS_' + year + '_30m', COUNT_NODATA);
  print(year + ' 年筛选后整景数量（不等于每像元有效数量）', scenes.size());
  Map.addLayer(annual, ndviVis, 'NDVI ' + year, year === years[years.length - 1]);
  Map.addLayer(nobs, {min: 0, max: 20, palette: ['d73027','fee08b','1a9850']},
    'Valid scene observations ' + year, false);
  Map.addLayer(nobs.lt(LOW_OBS_WARNING).selfMask(), {palette: ['ff00ff']},
    'Low observation coverage ' + year, false);

  var rows = scenes.toList(scenes.size()).map(function(obj) {
    var image = ee.Image(obj);
    return ee.Feature(null, {
      year: year, image_id: image.id(),
      product_id: image.get('LANDSAT_PRODUCT_ID'),
      scene_id: image.get('LANDSAT_SCENE_ID'),
      date: image.date().format('YYYY-MM-dd'),
      time_utc: image.date().format("YYYY-MM-dd'T'HH:mm:ss"),
      cloud_cover: image.get('CLOUD_COVER'),
      wrs_path: image.get('WRS_PATH'), wrs_row: image.get('WRS_ROW'),
      processing_level: image.get('PROCESSING_LEVEL')
    });
  });
  manifest = manifest.merge(ee.FeatureCollection(rows));
  yearSummaries.push(ee.Feature(null, {
    year: year, start_date: begin.format('YYYY-MM-dd'),
    end_date_exclusive: end.format('YYYY-MM-dd'), scene_count: scenes.size(),
    collection: 'LANDSAT/LC08/C02/T1_L2', composite: 'median_of_scene_NDVI',
    buffer_m_each_side: BUFFER_M, crs: CRS, grid_transform: GRID.join(','),
    scene_cloud_max_exclusive: SCENE_CLOUD_MAX, min_valid_observations: MIN_VALID_OBS,
    qa_pixel_clear_bits: '0,1,2,3,4,5', qa_radsat_rule: 'equals_zero',
    reflectance_rule: 'red_and_nir_nonnegative_sum_positive',
    scale_factor: 0.0000275, offset: -0.2,
    ndvi_nodata: NDVI_NODATA, nobs_nodata: COUNT_NODATA,
    pipeline_source_sha256: '3cc5e6bc248214e39367b245587f6f19f6d683bf418995285e8b2054e9ba3909'
  }));
});

Export.table.toDrive({
  collection: manifest, description: 'CN_RU_Landsat_scene_manifest_' + TAG,
  fileNamePrefix: 'CN_RU_Landsat_scene_manifest_' + TAG,
  folder: DRIVE_FOLDER, fileFormat: 'CSV',
  selectors: ['year','image_id','product_id','scene_id','date','time_utc',
    'cloud_cover','wrs_path','wrs_row','processing_level']
});
Export.table.toDrive({
  collection: ee.FeatureCollection(yearSummaries),
  description: 'CN_RU_annual_processing_log_' + TAG,
  fileNamePrefix: 'CN_RU_annual_processing_log_' + TAG,
  folder: DRIVE_FOLDER, fileFormat: 'CSV',
  selectors: ['year','start_date','end_date_exclusive','scene_count','collection',
    'composite','buffer_m_each_side','crs','grid_transform','scene_cloud_max_exclusive',
    'min_valid_observations','qa_pixel_clear_bits','qa_radsat_rule','reflectance_rule',
    'scale_factor','offset','ndvi_nodata','nobs_nodata','pipeline_source_sha256']
});
Map.addLayer(ee.FeatureCollection([ee.Feature(pipeline)]).style({
  color: 'ff0000', width: 2
}), {}, 'Accepted complete pipeline');
print('已创建导出任务数', years.length * 2 + 2);
print('请到右侧Tasks逐项点击RUN提交；编辑器Run只生成任务，不自动导出。');
print('TEST正常后，将RUN_ALL_YEARS改为true重新运行，即创建24个任务。');
