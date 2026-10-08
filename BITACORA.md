# Bitácora del Proyecto: Detección de Cobertura Terrestre en Colombia

## Información general

- **Fecha de inicio:** 2026-10-07
- **Objetivo:** Crear un programa en Python para detectar el tipo de cobertura terrestre de una zona de Colombia empleando modelos de IA geoespacial de código abierto.
- **Directorio de trabajo:** `C:\Users\Estudiantes\Documents\Taller`

---

## Prompts del usuario

### Prompt 1
> Crear un programa en python que me permita detectar el tipo de cobertura terrestre de una zona de Colombia empleando modelos de inteligencia artificial geoespacial de código abierto.

**Respuesta del asistente:** Exploración del directorio de trabajo (vacío, solo `.git/`). Se realizaron preguntas de clarificación al usuario.

### Prompt 2 (respuesta a preguntas de clarificación)
> - Fuente de datos: Sentinel-2 (ESA)
> - Modelo IA: Prithvi (NASA/HuggingFace)
> - Clases objetivo: Cobertura general (5-8 clases)
> - Datos etiquetados: No, usaré modelo pre-entrenado

**Respuesta del asistente:** Investigación web sobre Prithvi. Se propuso un plan detallado.

### Prompt 3
> Procede

**Respuesta del asistente:** El asistente indicó que estaba en modo planificación (solo lectura) y no podía crear archivos. Se solicitó desactivar el modo planificación.

### Prompt 4
> ya lo cambió

**Respuesta del asistente:** Se creó la estructura completa del proyecto.

### Prompt 5
> me sale esto para ejecutar [captura de pantalla con errores de directorio]

**Respuesta del asistente:** Se indicó que el usuario estaba en el directorio incorrecto y se proporcionaron los comandos `cd` correctos.

### Prompt 6
> crea un archivo de bitácora que consignes todas las decisiones y prompts utilizados

**Respuesta del asistente:** Creación del presente archivo.

---

## Decisiones técnicas

### 1. Modelo de IA seleccionado

| Aspecto | Decisión |
|---------|----------|
| Modelo | `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification` |
| Fuente | HuggingFace |
| Tipo | Vision Transformer (ViT) con patches 3D espacio-temporales |
| Pre-entrenamiento | Datos HLS (Harmonized Landsat Sentinel-2), 4.2M muestras |
| Ajuste fino | Clasificación de cultivos y coberturas con CDL (Crop Data Layer) de EE.UU. |
| Clases de salida | 13 clases (Natural Vegetation, Forest, Corn, Soybeans, Wetlands, Developed/Barren, Open Water, Winter Wheat, Alfalfa, Fallow/Idle Cropland, Cotton, Sorghum, Other) |

**Justificación:** Es el único modelo Prithvi con ajuste fino específico para clasificación de coberturas terrestres disponible en HuggingFace. No requiere datos etiquetados adicionales.

### 2. Fuente de imágenes satelitales

| Aspecto | Decisión |
|---------|----------|
| Fuente | Sentinel-2 L2A (ESA) |
| Catálogo | Copernicus Data Space Ecosystem |
| Resolución | 10m (bandas B02, B03, B04) y 20m (B8A, B11, B12) |
| Bandas utilizadas | B02 (Blue), B03 (Green), B04 (Red), B8A (Narrow NIR), B11 (SWIR 1), B12 (SWIR 2) |
| Pasos temporales | 3 (composites para reducir nubosidad) |

### 3. Mapeo de clases

| Clase Prithvi (CDL) | Cobertura general Colombia |
|---------------------|---------------------------|
| Forest | Bosque |
| Natural Vegetation | Vegetación natural |
| Corn, Soybeans, Winter Wheat, Alfalfa, Fallow/Idle Cropland, Cotton, Sorghum | Agricultura |
| Open Water | Agua |
| Developed/Barren | Urbano / Suelo desnudo |
| Wetlands | Humedales |
| Other | Otros |

### 4. Arquitectura del software

| Componente | Archivo | Responsabilidad |
|------------|---------|-----------------|
| Modelo | `land_cover_detection/model.py` | Carga del modelo Prithvi desde HuggingFace |
| Preprocesamiento | `land_cover_detection/preprocessing.py` | Descarga Sentinel-2, extracción de bandas, normalización |
| Inferencia | `land_cover_detection/inference.py` | Ejecución del modelo y post-procesamiento |
| Visualización | `land_cover_detection/visualization.py` | Generación de mapas y estadísticas |
| Utilidades | `land_cover_detection/utils.py` | Funciones auxiliares |
| CLI | `main.py` | Interfaz de línea de comandos |

### 5. Dependencias principales

- `torch>=2.0.0` — Framework de deep learning
- `transformers>=4.35.0` — Carga del modelo desde HuggingFace
- `rasterio>=1.3.8` — Manejo de rasters GeoTIFF
- `numpy>=1.24.0` — Computación numérica
- `matplotlib>=3.7.0` — Visualización
- `requests>=2.31.0` — Peticiones HTTP a Copernicus
- `geopandas>=0.13.0` — Manejo de datos geoespaciales
- `folium>=0.14.0` — Mapas interactivos

### 6. Regiones predefinidas de Colombia

| Región | Bounding box |
|--------|-------------|
| Amazonía | [-79.0, -4.5, -66.5, 1.5] |
| Andina | [-76.5, 0.5, -72.0, 7.5] |
| Caribe | [-77.5, 7.5, -71.0, 12.5] |
| Orinoquía | [-73.5, 3.5, -66.5, 8.5] |
| Pacífica | [-79.0, 1.0, -75.5, 7.5] |
| Antioquia | [-77.0, 5.5, -73.5, 9.0] |
| Cundinamarca | [-74.8, 3.8, -73.0, 5.8] |
| Valle del Cauca | [-77.5, 3.0, -75.5, 5.0] |

---

## Archivos creados

| Archivo | Descripción |
|---------|-------------|
| `land_cover_detection/__init__.py` | Inicialización del paquete |
| `land_cover_detection/model.py` | Clase `PrithviLandCoverModel` |
| `land_cover_detection/preprocessing.py` | Clase `Sentinel2Preprocessor` |
| `land_cover_detection/inference.py` | Clase `LandCoverInference` |
| `land_cover_detection/visualization.py` | Clase `LandCoverVisualizer` |
| `land_cover_detection/utils.py` | Funciones auxiliares |
| `main.py` | Script principal (CLI) |
| `requirements.txt` | Dependencias del proyecto |
| `README.md` | Documentación del proyecto |
| `BITACORA.md` | Presente archivo |

---

## Errores y correcciones

| Error | Causa | Corrección |
|-------|-------|------------|
| `Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'` | El usuario ejecutó el comando desde `C:\Users\Estudiantes>` en lugar de `C:\Users\Estudiantes\Documents\Taller` | Se indicó usar `cd C:\Users\Estudiantes\Documents\Taller` |
| `can't open file 'C:\\Users\\Estudiantes\\main.py': [Errno 2] No such file or directory` | Mismo error de directorio | Misma corrección |
| Import faltante de `torch` en `preprocessing.py` | Se usó `torch.from_numpy` sin importar `torch` | Se agregó `import torch` |

---

## Investigación web realizada

1. **Prithvi-EO-2.0 Paper** (arxiv:2412.02732) — Arquitectura y capacidades del modelo
2. **HuggingFace: Prithvi-EO-1.0-100M-multi-temporal-crop-classification** — Especificaciones del modelo ajustado
3. **GitHub: NASA-IMPACT/Prithvi-EO-2.0** — Repositorio oficial con ejemplos
4. **NASA Science** — Artículos sobre aplicaciones del modelo Prithvi

---

## Limitaciones identificadas

1. **Transferencia geográfica:** El modelo fue entrenado con datos de EE.UU. (CDL), por lo que su rendimiento en ecosistemas tropicales de Colombia puede ser limitado.
2. **Clases de cultivos:** Las clases específicas de EE.UU. (maíz, soya, trigo) no son relevantes para Colombia y se agrupan en "Agricultura".
3. **Resolución:** Limitada a 10-20m según la banda de Sentinel-2.
4. **Conectividad:** Se requiere conexión a internet para descargar imágenes y el modelo.
5. **Nubosidad:** Las nubes pueden afectar la calidad de la clasificación, especialmente en regiones tropicales.

---

## Posibles mejoras futuras

1. **Fine-tuning con datos de Colombia:** Ajustar el modelo con coberturas del IDEAM o IGAC.
2. **Prithvi-EO-2.0:** Usar la versión más reciente con embeddings temporales y de ubicación.
3. **Ensamblaje de modelos:** Combinar Prithvi con otros modelos para mejorar la precisión.
4. **Validación con datos de campo:** Comparar resultados con coberturas verificadas in situ.
5. **Interfaz web:** Crear una aplicación web para facilitar el uso.
