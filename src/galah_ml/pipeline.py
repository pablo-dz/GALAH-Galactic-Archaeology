"""Flujos de trabajo completos: construcción del catálogo y los tres experimentos con Random Forest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import models, plotting
from .data_loading import load_and_merge
from .preprocessing import (
    COMPONENT_COLUMNS,
    build_science_catalog,
    element_columns,
    label_components,
    oversample_class,
    remove_outliers,
)

STRUCTURE_TITLES = {"halo": "Halo", "thin_disk": "Thin Disk", "thick_disk": "Thick Disk"}
CLASSIFICATION_METRICS = {"precision": "Precision", "accuracy": "Accuracy", "recall": "Recall"}
REGRESSION_METRICS = {"r2": r"$\mathrm{R^2 Score}$", "rmse": "RMS", "mae": "MAE"}


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _fig_path(cfg: dict, name: str) -> Path:
    return cfg["paths"]["figures_dir"] / f"{name}.{cfg['figures']['format']}"


def _dpi(cfg: dict) -> int:
    return cfg["figures"]["dpi"]


def _ensure_dirs(cfg: dict) -> None:
    cfg["paths"]["figures_dir"].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["results_dir"].mkdir(parents=True, exist_ok=True)


def _load_ready(cfg: dict) -> pd.DataFrame:
    path = cfg["paths"]["processed_dir"] / cfg["catalog"]["output_file"]
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero scripts/01_build_catalog.py")
    return pd.read_csv(path)


def _save_json(cfg: dict, name: str, payload: dict) -> Path:
    out = cfg["paths"]["results_dir"] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


def _run_sweeps(cfg, kind, tag, sweeps, X_train, y_train, X_test, y_test, metric_labels) -> None:
    """Ejecuta los barridos de hiperparámetros descritos en la configuración y guarda tablas y figuras."""
    for sw in sweeps:
        param = sw["param"]
        values = np.arange(*sw["range"])
        df = models.sweep_hyperparameter(
            kind, X_train, y_train, X_test, y_test, param, values,
            base_params=sw.get("base"), n_jobs=cfg["n_jobs"], seed=cfg["random_seed"],
        )
        results_dir = cfg["paths"]["results_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / f"sweep_{tag}_{param}.csv", index=False)
        plotting.plot_metric_sweep(
            df, param, metric_labels, _fig_path(cfg, f"sweep_{tag}_{param}"), dpi=_dpi(cfg)
        )


# ---------------------------------------------------------------------------
# 1) Catálogo
# ---------------------------------------------------------------------------


def build_catalog(cfg: dict) -> Path:
    """Lee los catálogos FITS de GALAH DR4, aplica los cortes de calidad y guarda ``ready_data.csv``."""
    cat = cfg["catalog"]
    merged = load_and_merge(cfg["paths"]["raw_dir"], cat["files"])
    ready = build_science_catalog(
        merged, cat["base_columns"], snr_min=cat["snr_min"], flag_sp=cat["flag_sp"],
        max_nan_percent=cat["max_nan_percent"], excluded_elements=tuple(cat["excluded_elements"]),
        verbose=True,
    )
    out = cfg["paths"]["processed_dir"] / cat["output_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(out, index=False)
    print(f"Catálogo final: {len(ready):,} estrellas -> {out}")
    return out


# ---------------------------------------------------------------------------
# 2) Clasificación de componentes
# ---------------------------------------------------------------------------


def run_classification(cfg: dict, sweeps: bool = False) -> dict:
    """Diagrama de Toomre + un Random Forest binario por componente (halo, disco delgado, disco grueso)."""
    _ensure_dirs(cfg)
    ccfg, seed = cfg["classification"], cfg["random_seed"]
    sigma = cfg["outliers"]["sigma"]
    radii = (cfg["toomre"]["r_thin"], cfg["toomre"]["r_thick"], cfg["toomre"]["r_halo"])

    df = label_components(_load_ready(cfg), *radii)
    plotting.plot_toomre(df, radii, _fig_path(cfg, "toomre_diagram"), dpi=_dpi(cfg))
    plotting.plot_outlier_histograms(
        df, ccfg["outlier_plot_columns"], _fig_path(cfg, "outliers_distribution"), sigma, _dpi(cfg)
    )

    labeled = df[(df[COMPONENT_COLUMNS] == 1).any(axis=1)]
    excluded = set(ccfg["excluded_features"]) | set(COMPONENT_COLUMNS)
    features = [c for c in labeled.columns if c not in excluded]
    labeled = remove_outliers(labeled, features, sigma)

    bal = ccfg["balance"]
    counts = {c: int(labeled[c].sum()) for c in COMPONENT_COLUMNS}
    print("Estrellas por componente (sin balancear):", counts)
    if not bal["after_split"]:  # procedimiento del informe: balancear todo el conjunto antes de dividir
        labeled_eq = oversample_class(labeled, "halo", bal["halo_copies"])

    results = {"class_counts": counts, "balance_after_split": bal["after_split"], "models": {}}
    for target, tcfg in ccfg["targets"].items():
        if bal["after_split"]:  # alternativa: dividir primero y sobremuestrear solo el entrenamiento
            X_tr, X_te, y_tr, y_te = models.split_data(labeled[features], labeled[target], tcfg["test_size"], seed)
            train_df = oversample_class(labeled.loc[X_tr.index], "halo", bal["halo_copies"])
            X_tr, y_tr = train_df[features], train_df[target]
        else:
            X_tr, X_te, y_tr, y_te = models.split_data(
                labeled_eq[features], labeled_eq[target], tcfg["test_size"], seed
            )

        if sweeps:
            _run_sweeps(cfg, "classifier", target, ccfg["sweeps"].get(target, []), X_tr, y_tr, X_te, y_te,
                        CLASSIFICATION_METRICS)

        clf = models.train_classifier(X_tr, y_tr, n_jobs=cfg["n_jobs"], seed=seed, **tcfg["model"])
        scores = models.evaluate_classifier(clf, X_te, y_te)
        importances = models.feature_importances(clf, features)
        plotting.plot_confusion_matrix(
            models.confusion_matrix_normalized(clf, X_te, y_te), clf.classes_, STRUCTURE_TITLES[target],
            _fig_path(cfg, f"CM_{target}"), dpi=_dpi(cfg),
        )
        importances.to_csv(cfg["paths"]["results_dir"] / f"importances_{target}.csv", header=["importance"])
        results["models"][target] = {
            "hyperparameters": tcfg["model"], "test_size": tcfg["test_size"],
            "metrics": scores, "top_features": importances.head(5).round(4).to_dict(),
        }
        print(f"[{STRUCTURE_TITLES[target]}] " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    _save_json(cfg, "metrics_classification.json", results)
    return results


# ---------------------------------------------------------------------------
# 3) Predicción de [Mg/Fe]
# ---------------------------------------------------------------------------


def run_mg_fe(cfg: dict, sweeps: bool = False) -> dict:
    """Random Forest de regresión para [Mg/Fe] a partir del resto de abundancias y parámetros estelares."""
    _ensure_dirs(cfg)
    mcfg, seed = cfg["mg_fe"], cfg["random_seed"]
    df = _load_ready(cfg)
    features = [c for c in df.columns if c not in set(mcfg["excluded_features"])]
    data = remove_outliers(df, features, cfg["outliers"]["sigma"])

    X_tr, X_te, y_tr, y_te = models.split_data(data[features], data[mcfg["target"]], mcfg["test_size"], seed)
    if sweeps:
        _run_sweeps(cfg, "regressor", "mg_fe", mcfg["sweeps"], X_tr, y_tr, X_te, y_te, REGRESSION_METRICS)

    regr = models.train_regressor(X_tr, y_tr, n_jobs=cfg["n_jobs"], seed=seed, **mcfg["model"])
    scores = models.evaluate_regressor(regr, X_te, y_te)
    importances = models.feature_importances(regr, features)
    plotting.plot_true_vs_pred(y_te, regr.predict(X_te), "True vs Predicted Mg/Fe",
                               _fig_path(cfg, "mg_pre_obs"), dpi=_dpi(cfg))
    importances.to_csv(cfg["paths"]["results_dir"] / "importances_mg_fe.csv", header=["importance"])

    results = {"hyperparameters": mcfg["model"], "test_size": mcfg["test_size"], "n_stars": int(len(data)),
               "metrics": scores, "top_features": importances.head(5).round(4).to_dict()}
    print("[Mg/Fe] " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    _save_json(cfg, "metrics_mg_fe.json", results)
    return results


# ---------------------------------------------------------------------------
# 4) Reloj químico: edad a partir de abundancias
# ---------------------------------------------------------------------------


def run_age(cfg: dict, sweeps: bool = False) -> dict:
    """Random Forest de regresión para la edad estelar usando solo abundancias [X/Fe] (relojes químicos)."""
    _ensure_dirs(cfg)
    acfg, seed = cfg["age"], cfg["random_seed"]
    df = _load_ready(cfg)
    sel = df[(df["teff"] < acfg["teff_max"]) & (df["fe_h"] > acfg["fe_h_min"]) & (df["fe_h"] < acfg["fe_h_max"])]
    features = element_columns(sel)

    X_tr, X_te, y_tr, y_te = models.split_data(sel[features], sel[acfg["target"]], acfg["test_size"], seed)
    if sweeps:
        _run_sweeps(cfg, "regressor", "age", acfg["sweeps"], X_tr, y_tr, X_te, y_te, REGRESSION_METRICS)

    regr = models.train_regressor(X_tr, y_tr, n_jobs=cfg["n_jobs"], seed=seed, **acfg["model"])
    scores = models.evaluate_regressor(regr, X_te, y_te)
    importances = models.feature_importances(regr, features)
    plotting.plot_true_vs_pred(y_te, regr.predict(X_te), "True vs Predicted Age",
                               _fig_path(cfg, "clock_pre_obs"), dpi=_dpi(cfg))
    plot_cols = [c for c in acfg["abundance_age_columns"] if c in sel.columns]
    plotting.plot_abundance_vs_age(sel, plot_cols, _fig_path(cfg, "abundance_age"), dpi=_dpi(cfg))
    importances.to_csv(cfg["paths"]["results_dir"] / "importances_age.csv", header=["importance"])

    results = {"hyperparameters": acfg["model"], "test_size": acfg["test_size"], "n_stars": int(len(sel)),
               "metrics": scores, "top_features": importances.head(6).round(4).to_dict()}
    print("[Edad] " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    _save_json(cfg, "metrics_age.json", results)
    return results
