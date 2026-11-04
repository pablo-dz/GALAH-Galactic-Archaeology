# Datos

Esta carpeta no contiene datos (los catálogos son muy pesados). Estructura esperada:

```
data/
├── raw/         # catálogos FITS de GALAH DR4 descargados
└── processed/   # ready_data.csv, generado por scripts/01_build_catalog.py
```

## Catálogos necesarios (`data/raw/`)

| Archivo | Contenido |
|---|---|
| `galah_dr4_allstar_240705.fits` | Parámetros estelares, abundancias [X/Fe] y flags de calidad |
| `galah_dr4_vac_3dnlte_a_li_240705.fits` | Abundancia de litio (3D-NLTE) |
| `galah_dr4_vac_dynamics_240705.fits` | Edades, masas, velocidades U, V, W y órbitas |
| `galah_dr4_vac_wise_tmass_gaiadr3_240705.fits` | Cruce con WISE, 2MASS y Gaia DR3 |

Se obtienen desde la página oficial de GALAH DR4: https://www.galah-survey.org

Los nombres de archivo se pueden cambiar en `configs/config.yaml` (`catalog.files`).

## Catálogo procesado

`scripts/01_build_catalog.py` une los cuatro archivos por `sobject_id`, aplica los cortes de calidad y guarda `data/processed/ready_data.csv`, que contiene los parámetros estelares, las velocidades U, V, W y las abundancias [X/Fe] confiables.
