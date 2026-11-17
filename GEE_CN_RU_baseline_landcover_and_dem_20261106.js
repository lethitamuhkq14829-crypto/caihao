var CRS = 'EPSG:32652';
var GRID = [30, 0, 0, 0, -30, 10000020];
var CORRIDOR_ASSET = 'REPLACE_WITH_YOUR_ASSET_ID';
if (CORRIDOR_ASSET.indexOf('REPLACE_') === 0) throw new Error('Set CORRIDOR_ASSET before running.');
var EXPORT_BOUNDS_UTM = [291180, 5350590, 454320, 5726580];

var DRIVE_FOLDER = 'CN_RU_Baseline_Covariates_20261106';
var LANDCOVER_NODATA = 255;
var DEM_NODATA = -9999;
var corridor = ee.FeatureCollection(CORRIDOR_ASSET).geometry(1);
var exportRegion = ee.Geometry.Rectangle(EXPORT_BOUNDS_UTM, CRS, false);

var annualLandCover = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual')
  .filterBounds(exportRegion);

function landCoverForYear(year) {
  var band = 'b' + (year - 2000 + 1);
  return annualLandCover.select(band).map(function (tile) {
    var code = tile.select(band);
    return code.updateMask(code.gt(0).and(code.lte(220)));
  }).mosaic().rename('land_cover').toUint8().clip(corridor);
}

function exportGeoTiff(image, filename, noData) {
  Export.image.toDrive({
    image: image.unmask({value: noData, sameFootprint: false}),
    description: filename,
    folder: DRIVE_FOLDER,
    fileNamePrefix: filename,
    region: exportRegion,
    crs: CRS,
    crsTransform: GRID,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: noData}
  });
}

var lc2013 = landCoverForYear(2013);
var lc2014 = landCoverForYear(2014);
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
  .toFloat().resample('bilinear').clip(corridor);

exportGeoTiff(lc2013, 'CN_RU_GLC_FCS30D_2013_30m', LANDCOVER_NODATA);
exportGeoTiff(lc2014, 'CN_RU_GLC_FCS30D_2014_30m', LANDCOVER_NODATA);
exportGeoTiff(dem, 'CN_RU_SRTM_DEM_30m', DEM_NODATA);

Map.centerObject(exportRegion, 7);
Map.addLayer(corridor, {color: '00a5cc'}, 'NDVI corridor', false);
Map.addLayer(lc2013, {min: 10, max: 220}, '2013 land-cover codes', false);
Map.addLayer(dem, {min: 100, max: 900}, 'SRTM elevation (m)', false);
print('GLC_FCS30D tiles intersecting the export region', annualLandCover.size());
print('Land-cover bands: 2013 = b14, 2014 = b15');
print('Drive folder', DRIVE_FOLDER);
print('Expected: 3 export tasks, 5438 columns x 12533 rows each');