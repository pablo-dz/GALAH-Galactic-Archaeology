"""Datos sintéticos con la misma estructura que el catálogo de GALAH (solo para pruebas)."""
import numpy as np
import pandas as pd

ELEMENTS = ["c", "n", "o", "na", "mg", "al", "si", "k", "ca", "ti", "y", "ba"]


def make_ready_catalog(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comp = rng.choice(["thin", "thick", "halo"], size=n, p=[0.55, 0.43, 0.02])
    speed = {"thin": 25.0, "thick": 110.0, "halo": 260.0}
    v_tot = np.array([rng.normal(speed[c], 8.0) for c in comp]).clip(1)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    uvw = direction * v_tot[:, None]

    age = rng.uniform(0.5, 12, n)
    fe_h = rng.normal(-0.1, 0.25, n) - 0.01 * age
    data = {
        "sobject_id": np.arange(n),
        "teff": rng.normal(5300, 350, n), "logg": rng.normal(2.5, 0.8, n), "fe_h": fe_h,
        "vmic": rng.normal(1.2, 0.2, n), "vsini": rng.normal(5, 1, n), "mass": rng.normal(1.1, 0.2, n),
        "age": age, "lbol": rng.normal(50, 10, n),
        "U_UVW": uvw[:, 0], "V_UVW": uvw[:, 1], "W_UVW": uvw[:, 2],
    }
    mg = 0.02 * age - 0.3 * fe_h + rng.normal(0, 0.03, n)
    for el in ELEMENTS:
        data[f"{el}_fe"] = rng.normal(0, 0.08, n)
    data["mg_fe"] = mg
    data["si_fe"] = 0.5 * mg + rng.normal(0, 0.03, n)
    data["y_fe"] = -0.02 * age + rng.normal(0, 0.05, n)
    return pd.DataFrame(data)


def make_merged_catalog(n: int = 500, seed: int = 1) -> pd.DataFrame:
    """Tabla con la forma de la unión de los catálogos, incluyendo flags y NaN."""
    rng = np.random.default_rng(seed)
    df = make_ready_catalog(n, seed)
    df["snr_px_ccd3"] = rng.uniform(10, 100, n)
    df["flag_sp"] = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    for el in ELEMENTS:
        df[f"flag_{el}_fe"] = rng.choice([0, 1], size=n, p=[0.95, 0.05])
        df[f"e_{el}_fe"] = rng.uniform(0, 0.1, n)
    df["eu_fe"] = np.where(rng.random(n) < 0.6, np.nan, 0.0)  # abundancia con demasiados NaN
    df["flag_eu_fe"] = 0
    df["nn_li_fe"] = rng.normal(size=n)
    df["flag_nn_li_fe"] = 0
    return df
