"""Prueba de humo de los tres experimentos con datos sintéticos y modelos pequeños."""
import copy
import json

import pytest

from galah_ml import pipeline, plotting
from galah_ml.config import load_config

from synthetic import make_ready_catalog


@pytest.fixture()
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {k: tmp_path / k for k in ("raw_dir", "processed_dir", "figures_dir", "results_dir")}
    cfg["figures"] = {"format": "png", "dpi": 50}
    cfg["n_jobs"] = 1
    for t in cfg["classification"]["targets"].values():
        t["model"] = {"n_estimators": 10, "max_depth": 6}
    cfg["classification"]["sweeps"] = {k: [{"param": "max_depth", "range": [1, 4, 1]}] for k in ("halo", "thin_disk")}
    cfg["mg_fe"]["model"] = {"n_estimators": 10, "max_depth": 6}
    cfg["mg_fe"]["sweeps"] = [{"param": "max_depth", "range": [1, 4, 1]}]
    cfg["age"]["model"] = {"n_estimators": 10, "max_depth": 6}
    cfg["age"]["sweeps"] = [{"param": "max_depth", "range": [1, 4, 1]}]
    (tmp_path / "processed_dir").mkdir()
    make_ready_catalog(3000).to_csv(tmp_path / "processed_dir" / cfg["catalog"]["output_file"], index=False)
    plotting.set_plot_style()
    return cfg


def test_classification(cfg):
    res = pipeline.run_classification(cfg, sweeps=True)
    assert set(res["models"]) == {"halo", "thin_disk", "thick_disk"}
    assert (cfg["paths"]["figures_dir"] / "CM_halo.png").exists()
    assert (cfg["paths"]["figures_dir"] / "toomre_diagram.png").exists()
    assert (cfg["paths"]["results_dir"] / "metrics_classification.json").exists()


def test_classification_balance_after_split(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["classification"]["balance"]["after_split"] = True
    res = pipeline.run_classification(cfg)
    assert res["balance_after_split"] is True
    assert 0 <= res["models"]["halo"]["metrics"]["recall"] <= 1


def test_mg_fe(cfg):
    res = pipeline.run_mg_fe(cfg, sweeps=True)
    assert res["metrics"]["r2"] > 0.3
    assert (cfg["paths"]["figures_dir"] / "mg_pre_obs.png").exists()
    assert json.load(open(cfg["paths"]["results_dir"] / "metrics_mg_fe.json"))["n_stars"] > 0


def test_age(cfg):
    res = pipeline.run_age(cfg, sweeps=True)
    assert "y_fe" in res["top_features"] or "mg_fe" in res["top_features"]
    assert (cfg["paths"]["figures_dir"] / "abundance_age.png").exists()
