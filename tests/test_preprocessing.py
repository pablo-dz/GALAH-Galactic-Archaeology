import numpy as np
import pandas as pd

from galah_ml import preprocessing as pp
from galah_ml.data_loading import merge_catalogs

from synthetic import make_merged_catalog, make_ready_catalog

BASE = ["sobject_id", "teff", "logg", "fe_h", "vmic", "vsini", "mass", "age", "lbol", "U_UVW", "V_UVW", "W_UVW"]


def test_label_components_thresholds():
    df = pd.DataFrame({"U_UVW": [10, 0, 0, 0, 0], "V_UVW": [0, 60, 100, 200, 0], "W_UVW": [0, 0, 0, 0, 300]})
    out = pp.label_components(df)
    assert out["thin_disk"].tolist() == [1, 0, 0, 0, 0]
    assert out["thick_disk"].tolist() == [0, 0, 1, 0, 0]
    assert out["halo"].tolist() == [0, 0, 0, 1, 1]
    # 60 km/s cae en la zona de traslape: sin clasificar
    assert out.loc[1, ["thin_disk", "thick_disk", "halo"]].sum() == 0


def test_sigma_clip_mask_flags_extreme_values():
    rng = np.random.default_rng(0)
    x = np.append(rng.normal(0, 1, 1000), [50.0, -60.0])
    mask = pp.sigma_clip_mask(x, sigma=5)
    assert mask[-2:].all()
    assert mask.sum() < 10


def test_remove_outliers_drops_rows():
    df = make_ready_catalog(500)
    df.loc[0, "teff"] = 1e6
    out = pp.remove_outliers(df, ["teff", "logg"], sigma=5)
    assert 0 not in out.index
    assert len(out) >= len(df) - 5


def test_oversample_class_adds_copies():
    df = pd.DataFrame({"halo": [1, 0, 0, 0], "x": [1, 2, 3, 4]})
    out = pp.oversample_class(df, "halo", copies=10)
    assert len(out) == 4 + 10
    assert out["halo"].sum() == 11
    assert pp.oversample_class(df, "halo", copies=0).shape == df.shape


def test_build_science_catalog_filters_quality_and_nans():
    merged = make_merged_catalog()
    ready = pp.build_science_catalog(merged, BASE, snr_min=30, flag_sp=0, max_nan_percent=15)
    assert "eu_fe" not in ready.columns          # demasiados NaN
    assert "nn_li_fe" not in ready.columns       # excluida explícitamente
    assert not ready.isna().any().any()
    assert len(ready) < len(merged)
    assert set(BASE).issubset(ready.columns)


def test_element_columns_ignore_flags_and_errors():
    df = make_merged_catalog()
    cols = pp.element_columns(df)
    assert "mg_fe" in cols
    assert not any(c.startswith(("flag_", "e_")) for c in cols)


def test_merge_catalogs_drops_duplicated_columns():
    a = pd.DataFrame({"sobject_id": [1, 2], "teff": [5000, 5500]})
    b = pd.DataFrame({"sobject_id": [1, 2], "teff": [1, 1], "li_fe": [0.1, 0.2]})
    c = pd.DataFrame({"sobject_id": [1, 2], "U_UVW": [1.0, 2.0]})
    d = pd.DataFrame({"sobject_id": [1, 2], "ra": [10.0, 20.0]})
    out = merge_catalogs(a, b, c, d)
    assert out["teff"].tolist() == [5000, 5500]
    assert {"li_fe", "U_UVW", "ra"}.issubset(out.columns)
