"""Preparación de datos: filtros de calidad, selección de columnas, etiquetas cinemáticas, outliers y balanceo."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Construcción del catálogo "listo para entrenar"
# ---------------------------------------------------------------------------


def quality_mask(df: pd.DataFrame, snr_min: float = 30, flag_sp: int = 0) -> pd.Series:
    """Máscara de calidad espectroscópica: ``snr_px_ccd3 > snr_min`` y ``flag_sp == flag_sp``."""
    return (df["snr_px_ccd3"] > snr_min) & (df["flag_sp"] == flag_sp)


def element_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de abundancias [X/Fe] (terminan en ``_fe``), excluyendo errores y flags."""
    excluded_prefixes = ("flag_", "e_", "nn_e", "nn_flag")
    return [c for c in df.columns if c.endswith("_fe") and not c.startswith(excluded_prefixes)]


def element_flag_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de flags de calidad asociadas a abundancias [X/Fe]."""
    return [c for c in df.columns if c.startswith(("flag_", "nn_flag")) and c.endswith("_fe")]


def build_science_catalog(
    merged: pd.DataFrame,
    base_columns: list[str],
    snr_min: float = 30,
    flag_sp: int = 0,
    max_nan_percent: float = 15.0,
    excluded_elements: tuple[str, ...] = ("nn_li_fe",),
    verbose: bool = False,
) -> pd.DataFrame:
    """Aplica los cortes de calidad y devuelve el catálogo final (parámetros + abundancias confiables).

    Pasos:
      1. Corte por ``snr_px_ccd3`` y ``flag_sp``.
      2. Descarta las abundancias con más de ``max_nan_percent`` % de NaN.
      3. Elimina filas con NaN en las columnas restantes.
      4. Conserva solo las estrellas con ``flag_X_fe == 0`` en todas las abundancias seleccionadas.
    """
    cols_elements = element_columns(merged)
    flag_cols = element_flag_columns(merged)
    sel = merged[quality_mask(merged, snr_min, flag_sp)][base_columns + cols_elements + flag_cols]

    selected, selected_flags = [], []
    for col in cols_elements:
        pct = 100 * sel[col].isna().sum() / len(sel[col])
        if verbose:
            print(f"{col} {pct:.2f} %")
        if pct < max_nan_percent and col not in excluded_elements:
            selected.append(col)
            selected_flags.append("flag_" + col)

    sel = sel[base_columns + selected + selected_flags].dropna()
    reliable = sel[(sel[selected_flags] == 0).all(axis=1)]
    return reliable[base_columns + selected].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Clasificación cinemática (diagrama de Toomre)
# ---------------------------------------------------------------------------

COMPONENT_COLUMNS = ["thin_disk", "thick_disk", "halo"]


def label_components(
    df: pd.DataFrame, r_thin: float = 50, r_thick: float = 70, r_halo: float = 180
) -> pd.DataFrame:
    """Etiqueta cada estrella como disco delgado, grueso o halo según su velocidad total.

    V_tot = sqrt(U² + V² + W²). Disco delgado: V_tot < r_thin; disco grueso: r_thick <= V_tot < r_halo;
    halo: V_tot >= r_halo. Las estrellas entre r_thin y r_thick quedan sin clasificar (zona de traslape).
    """
    out = df.copy()
    out["distance"] = np.sqrt(out["U_UVW"] ** 2 + out["V_UVW"] ** 2 + out["W_UVW"] ** 2)
    out["thin_disk"] = (out["distance"] < r_thin).astype(int)
    out["thick_disk"] = ((out["distance"] >= r_thick) & (out["distance"] < r_halo)).astype(int)
    out["halo"] = (out["distance"] >= r_halo).astype(int)
    return out


# ---------------------------------------------------------------------------
# Outliers y balanceo de clases
# ---------------------------------------------------------------------------


def sigma_clip_mask(x: np.ndarray, sigma: float = 5.0, maxiters: int = 5) -> np.ndarray:
    """Máscara booleana (True = outlier) con *sigma clipping* iterativo en torno a la mediana.

    Equivalente a ``astropy.stats.sigma_clip`` con sus valores por defecto (centro: mediana,
    dispersión: desviación estándar, hasta ``maxiters`` iteraciones).
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isfinite(x)
    for _ in range(maxiters):
        valid = x[~mask]
        if valid.size == 0:
            break
        center, std = np.median(valid), valid.std()
        new_mask = mask | (np.abs(x - center) > sigma * std)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    return mask


def iter_sigma_clipping(
    df: pd.DataFrame, columns: list[str], sigma: float = 5.0
) -> Iterator[tuple[str, pd.Series, np.ndarray]]:
    """Recorre ``columns`` aplicando sigma clipping de forma acumulativa.

    En cada paso devuelve (columna, valores antes del recorte, máscara de outliers) y deja que el
    DataFrame se reduzca para la columna siguiente.
    """
    current = df
    for col in columns:
        x = current[col]
        mask = sigma_clip_mask(x.to_numpy(), sigma=sigma)
        yield col, x, mask
        current = current[~mask]


def remove_outliers(df: pd.DataFrame, columns: list[str], sigma: float = 5.0) -> pd.DataFrame:
    """Elimina las filas que sean outliers (> ``sigma``) en alguna de ``columns``, columna por columna."""
    current = df
    for col in columns:
        mask = sigma_clip_mask(current[col].to_numpy(), sigma=sigma)
        current = current[~mask]
    return current


def oversample_class(df: pd.DataFrame, column: str, copies: int = 10) -> pd.DataFrame:
    """Añade ``copies`` copias de las filas con ``df[column] == 1`` (sobremuestreo de la clase minoritaria)."""
    if copies <= 0:
        return df.reset_index(drop=True)
    minority = df[df[column] == 1]
    return pd.concat([df] + [minority] * copies, ignore_index=True)
