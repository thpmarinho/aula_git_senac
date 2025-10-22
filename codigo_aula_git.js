// ===== 1. IMPORTAR SHAPES =====
// Treinamento, teste e área de recorte (todos devem estar no painel esquerdo)
var treino = ee.FeatureCollection("users/seu_usuario/treino");
var teste = ee.FeatureCollection("users/seu_usuario/teste");
var area = ee.FeatureCollection("users/seu_usuario/area");

// ===== 2. DEFINIR INTERVALO DE DATAS =====
var startDate = '2022-06-01';
var endDate = '2022-09-30';

// ===== 3. MÁSCARA DE NUVENS PARA SENTINEL-2 =====
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
              .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start']);
}

// ===== 4. COLEÇÃO SENTINEL-2 =====
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(area)  // agora filtrando pela área
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2clouds)
  .median()
  .clip(area);  // recorte final pela área

// ===== 5. BANDAS PARA CLASSIFICAÇÃO =====
var bands = ['B2', 'B3', 'B4', 'B8'];  // azul, verde, vermelho, NIR

// ===== 6. AMOSTRAGEM DE TREINO =====
var training = s2.select(bands).sampleRegions({
  collection: treino,
  properties: ['class'],
  scale: 10
});

// ===== 7. TREINAMENTO - CLASSIFICADOR CART =====
var classifier = ee.Classifier.smileCart().train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// ===== 8. CLASSIFICAÇÃO =====
var classified = s2.select(bands).classify(classifier).clip(area);

// ===== 9. VALIDAÇÃO COM TESTE =====
var validation = s2.select(bands).sampleRegions({
  collection: teste,
  properties: ['class'],
  scale: 10
});
var validated = validation.classify(classifier);
var confusionMatrix = validated.errorMatrix('class', 'classification');
print('Matriz de confusão:', confusionMatrix);
print('Acurácia geral:', confusionMatrix.accuracy());

// ===== 10. VISUALIZAÇÃO =====
var palette = ['red', 'green', 'blue', 'yellow', 'purple']; // ajuste conforme suas classes
Map.centerObject(area, 11);
Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'Imagem Sentinel-2');
Map.addLayer(classified, {min: 0, max: 4, palette: palette}, 'Classificação');
Map.addLayer(treino, {color: 'white'}, 'Treinamento');
Map.addLayer(teste, {color: 'black'}, 'Validação');
Map.addLayer(area, {color: 'cyan'}, 'Área de recorte');

// ===== 11. EXPORTAÇÃO (OPCIONAL) =====
/*
Export.image.toDrive({
  image: classified,
  description: 'classificacao_CART_com_area',
  region: area.geometry(),
  scale: 10,
  maxPixels: 1e13
});
*/
//alteracao v