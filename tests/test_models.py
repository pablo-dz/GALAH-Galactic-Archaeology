from galah_ml import models
from galah_ml.preprocessing import label_components

from synthetic import make_ready_catalog


def test_classifier_separates_halo():
    df = label_components(make_ready_catalog(3000))
    feats = ["teff", "logg", "fe_h", "mg_fe", "si_fe"]
    # el halo sintético no se distingue por química, así que usamos la velocidad solo para comprobar el flujo
    feats_v = feats + ["distance"]
    X_tr, X_te, y_tr, y_te = models.split_data(df[feats_v], df["halo"], 0.3, seed=0)
    clf = models.train_classifier(X_tr, y_tr, n_estimators=30, max_depth=6, n_jobs=1, seed=0)
    scores = models.evaluate_classifier(clf, X_te, y_te)
    assert set(scores) == {"accuracy", "precision", "recall"}
    assert scores["accuracy"] > 0.95
    cm = models.confusion_matrix_normalized(clf, X_te, y_te)
    assert abs(cm.sum() - 1) < 1e-9


def test_regressor_and_importances():
    df = make_ready_catalog(2000)
    feats = ["age", "fe_h", "teff", "si_fe"]
    X_tr, X_te, y_tr, y_te = models.split_data(df[feats], df["mg_fe"], 0.3, seed=0)
    regr = models.train_regressor(X_tr, y_tr, n_estimators=30, max_depth=8, n_jobs=1, seed=0)
    scores = models.evaluate_regressor(regr, X_te, y_te)
    assert scores["r2"] > 0.5
    imp = models.feature_importances(regr, feats)
    assert abs(imp.sum() - 1) < 1e-6
    assert imp.is_monotonic_decreasing


def test_sweep_returns_one_row_per_value():
    df = make_ready_catalog(1000)
    feats = ["age", "fe_h"]
    X_tr, X_te, y_tr, y_te = models.split_data(df[feats], df["mg_fe"], 0.3, seed=0)
    out = models.sweep_hyperparameter(
        "regressor", X_tr, y_tr, X_te, y_te, "max_depth", [1, 3, 5], base_params={"n_estimators": 10}, n_jobs=1
    )
    assert out["max_depth"].tolist() == [1, 3, 5]
    assert {"r2", "rmse", "mae"}.issubset(out.columns)
