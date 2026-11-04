"""Lectura de los catálogos FITS de GALAH DR4 y unión en una sola tabla."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_fits_table(path: str | Path, hdu: int = 1) -> pd.DataFrame:
    """Lee una tabla FITS (extensión ``hdu``) como DataFrame con orden de bytes nativo."""
    from astropy.io import fits  # import local: solo se necesita para leer FITS

    with fits.open(path) as hdul:
        data = hdul[hdu].data
        columns = {}
        for name in data.names:
            col = np.asarray(data[name])
            if col.ndim > 1:  # columnas vectoriales no caben en una tabla plana
                continue
            if col.dtype.byteorder not in ("=", "|"):
                col = col.astype(col.dtype.newbyteorder("="))
            columns[name] = col
    return pd.DataFrame(columns)


def merge_catalogs(
    allstar: pd.DataFrame,
    li_abundances: pd.DataFrame,
    dynamics: pd.DataFrame,
    gaia: pd.DataFrame,
    key: str = "sobject_id",
) -> pd.DataFrame:
    """Une los cuatro catálogos por ``key`` y descarta las columnas duplicadas de las tablas secundarias."""
    suffixes = ("_df_li_abundances", "_df_dynamics", "_df_gaia")
    merged = pd.merge(allstar, li_abundances, on=key, suffixes=("", suffixes[0]))
    merged = pd.merge(merged, dynamics, on=key, suffixes=("", suffixes[1]))
    merged = pd.merge(merged, gaia, on=key, suffixes=("", suffixes[2]))
    return merged.loc[:, ~merged.columns.str.endswith(suffixes)]


def load_and_merge(raw_dir: str | Path, files: dict[str, str]) -> pd.DataFrame:
    """Lee los cuatro archivos FITS indicados en ``files`` (claves: allstar, li, dynamics, gaia) y los une."""
    raw_dir = Path(raw_dir)
    tables = {name: read_fits_table(raw_dir / fname) for name, fname in files.items()}
    return merge_catalogs(tables["allstar"], tables["li"], tables["dynamics"], tables["gaia"])
