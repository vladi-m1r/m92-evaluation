# Localización histórica de archivos (M9.2)

Herramientas para localizar archivos Python de Django relacionados con tickets. El modo predeterminado reconstruye el commit padre de cada ticket para evitar buscar en código posterior al cambio.

## Requisitos

- Python y dependencias de `requirements.txt`.
- Un clon Git local de Django que contenga los commits padre indicados en el CSV.
- Para CUDA, una instalación de PyTorch compatible con la GPU y sus controladores. También se puede ejecutar en CPU.

Instala las dependencias:

```powershell
python -m pip install -r requirements.txt
```

La primera ejecución puede descargar el modelo multilingüe `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Generar predicciones

El CSV de entrada debe incluir `id`, `summary` y, para el modo histórico, `padre` (hash del commit padre). El dataset incluido es `dataset_20/dataset_20_tickets.csv`.

```powershell
python .\m92_localizacion_historica.py --repo "D:\ruta\al\clon\django" --input .\dataset_20\dataset_20_tickets.csv --out .\resultados_m92
```

El modo predeterminado es `--snapshot historical` y el dispositivo `--device auto` (CUDA si está disponible; si no, CPU). Opciones útiles:

- `--snapshot current`: usa el estado actual del clon, no el histórico.
- `--ticket ID`: procesa un solo ticket.
- `--top N`: cantidad máxima de archivos por ranking (20 por defecto).
- `--device auto|cuda|cpu`: selecciona el dispositivo para embeddings.

El script genera:

- `predicciones_m92.csv`: ranking final Weighted RRF.
- `predicciones_comparativas.csv`: rankings BM25, semántico y Weighted RRF.
- `configuracion_m92.json`: parámetros y metadatos de la ejecución.

## Evaluar Recall@K

El dataset debe tener las columnas `id` y `archivos_codigo`; las predicciones deben incluir `ticket`, `archivo` y `rank`. Para comparar los tres métodos, usa `predicciones_comparativas.csv`:

```powershell
python .\evaluar_recall.py --dataset .\dataset_20\dataset_20_tickets.csv --predictions .\resultados_m92\predicciones_comparativas.csv --out .\evaluacion_recall --ks 5 10 20
```

Si el CSV incluye `particion`, selecciona explícitamente el conjunto que quieras evaluar, por ejemplo:

```powershell
python .\evaluar_recall.py --dataset .\dataset_20\dataset_20_tickets.csv --predictions .\resultados_m92\predicciones_comparativas.csv --out .\evaluacion_prueba --partition prueba
```

La evaluación produce archivos `_detalle.csv` y `_resumen.csv` con resultados por ticket y métricas macro por ticket y micro por archivo. Sin `--partition`, se evalúan todas las filas.
