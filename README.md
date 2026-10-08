# Detección de Cobertura Terrestre en Colombia con Prithvi

Programa en Python para clasificar la cobertura terrestre de zonas de Colombia utilizando el modelo de inteligencia artificial geoespacial **Prithvi** (NASA/IBM) e imágenes **Sentinel-2** (ESA).

## Modelo

- **Prithvi-EO-1.0-100M-multi-temporal-crop-classification** ([HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification))
- Modelo de fundación geoespacial pre-entrenado con datos HLS (Harmonized Landsat Sentinel-2)
- Ajustado para clasificación de coberturas y cultivos con 13 clases
- Arquitectura: Vision Transformer (ViT) con patches 3D espacio-temporales

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso

### Ver regiones predefinidas

```bash
python main.py --list-regions
```

### Clasificar una región predefinida

```bash
python main.py --region "Amazonía" --start-date 2024-01-01 --end-date 2024-12-31
```

### Clasificar con bounding box personalizado

```bash
python main.py --bbox -74.5 4.0 -73.5 5.0 --start-date 2024-01-01 --end-date 2024-12-31
```

### Clasificar desde un archivo GeoJSON

```bash
python main.py --geojson zona_estudio.geojson --start-date 2024-01-01 --end-date 2024-12-31
```

### Especificar dispositivo y nubosidad

```bash
python main.py --region "Andina" --device cuda --cloud-cover 10 --output-dir resultados_andina
```

## Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--bbox` | Bounding box: min_lon min_lat max_lon max_lat | - |
| `--geojson` | Ruta a GeoJSON con zona de estudio | - |
| `--region` | Región predefinida de Colombia | - |
| `--start-date` | Fecha inicio (YYYY-MM-DD) | 2024-01-01 |
| `--end-date` | Fecha fin (YYYY-MM-DD) | 2024-12-31 |
| `--cloud-cover` | Nubosidad máxima (%) | 20 |
| `--output-dir` | Directorio de salida | output |
| `--device` | Dispositivo (cuda/cpu) | auto |
| `--list-regions` | Listar regiones disponibles | - |

## Salida

El programa genera los siguientes archivos en el directorio de salida:

- `land_cover_classification.tif` — Raster GeoTIFF con la clasificación
- `land_cover_map.png` — Mapa visual de coberturas
- `land_cover_statistics.png` — Gráfico de barras con porcentajes
- `land_cover_statistics.json` — Estadísticas en formato JSON

## Clases de cobertura

| Cobertura | Color |
|-----------|-------|
| Bosque | Verde oscuro |
| Vegetación natural | Verde claro |
| Agricultura | Amarillo |
| Agua | Azul |
| Urbano / Suelo desnudo | Gris |
| Humedales | Turquesa |
| Otros | Marrón |

## Estructura del proyecto

```
Taller/
├── land_cover_detection/
│   ├── __init__.py
│   ├── model.py              # Carga del modelo Prithvi
│   ├── preprocessing.py      # Descarga y preprocesamiento Sentinel-2
│   ├── inference.py          # Inferencia y post-procesamiento
│   ├── visualization.py      # Visualización de resultados
│   └── utils.py              # Utilidades
├── main.py                   # Script principal (CLI)
├── requirements.txt
└── README.md
```

## Limitaciones

- El modelo fue entrenado principalmente con datos de EE.UU. (CDL), por lo que su rendimiento en ecosistemas tropicales de Colombia puede ser limitado.
- Las clases de cultivos específicos de EE.UU. (maíz, soya, trigo) se agrupan en "Agricultura".
- La resolución espacial está limitada a 10m (resolución nativa de Sentinel-2).
- Se requiere conexión a internet para descargar imágenes Sentinel-2 y el modelo.

## Referencias

- [Prithvi-EO-2.0 Paper](https://arxiv.org/abs/2412.02732)
- [Prithvi en HuggingFace](https://huggingface.co/ibm-nasa-geospatial)
- [NASA Science - Prithvi](https://science.nasa.gov/science-research/ai-geospatial-model-earth/)
- [Copernicus Data Space](https://dataspace.copernicus.eu/)
