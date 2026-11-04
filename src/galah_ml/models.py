"""Entrenamiento y evaluación de Random Forest (clasificación y regresión) y barridos de hiperparámetros."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """División entrenamiento/prueba reproducible: devuelve X_train, X_test, y_train, y_test."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------


def train_classifier(
    X_train, y_train, n_estimators=200, max_depth=10, min_samples_split=2,
    min_samples_leaf=1, ccp_alpha=0.0, n_jobs=-1, seed=0,
) -> RandomForestClassifier:
    """Random Forest de clasificación binaria con ``class_weight='balanced'``."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators, criterion="gini", max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha, class_weight="balanced", n_jobs=n_jobs, random_state=seed,
    )
    return clf.fit(X_train, y_train)


def evaluate_classifier(clf, X_test, y_test) -> dict[str, float]:
    """Accuracy, precision y recall sobre el conjunto de prueba."""
    y_pred = clf.predict(X_test)
    return {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_test, y_pred, zero_division=0),
    }


def confusion_matrix_normalized(clf, X_test, y_test) -> np.ndarray:
    """Matriz de confusión normalizada sobre el total de muestras."""
    return metrics.confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_, normalize="all")


# ---------------------------------------------------------------------------
# Regresión
# ---------------------------------------------------------------------------


def train_regressor(
    X_train, y_train, n_estimators=200, max_depth=10, min_samples_split=2,
    min_samples_leaf=1, ccp_alpha=0.0, n_jobs=-1, seed=0,
) -> RandomForestRegressor:
    """Random Forest de regresión."""
    regr = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha, n_jobs=n_jobs, random_state=seed,
    )
    return regr.fit(X_train, y_train)


def evaluate_regressor(regr, X_test, y_test) -> dict[str, float]:
    """R², RMSE y MAE sobre el conjunto de prueba."""
    y_pred = regr.predict(X_test)
    return {
        "r2": metrics.r2_score(y_test, y_pred),
        "rmse": metrics.root_mean_squared_error(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
    }


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------


def feature_importances(model, columns: Iterable[str], min_importance: float = 0.0) -> pd.Series:
    """Importancia de cada variable (reducción de impureza), ordenada de mayor a menor."""
    s = pd.Series(model.feature_importances_, index=list(columns)).sort_values(ascending=False)
    return s[s > min_importance]


def sweep_hyperparameter(
    kind: str, X_train, y_train, X_test, y_test, param: str, values: Iterable,
    base_params: dict | None = None, n_jobs: int = -1, seed: int = 0,
) -> pd.DataFrame:
    """Varía un hiperparámetro (los demás quedan fijos) y devuelve las métricas para cada valor.

    ``kind`` es ``"classifier"`` o ``"regressor"``; ``param`` es un nombre de scikit-learn
    (``max_depth``, ``n_estimators``, ``min_samples_split``, ``min_samples_leaf``).
    """
    if kind == "classifier":
        train, evaluate = train_classifier, evaluate_classifier
    elif kind == "regressor":
        train, evaluate = train_regressor, evaluate_regressor
    else:
        raise ValueError("kind debe ser 'classifier' o 'regressor'")
    rows = []
    for v in values:
        params = {**(base_params or {}), param: v}
        model = train(X_train, y_train, n_jobs=n_jobs, seed=seed, **params)
        rows.append({param: v, **evaluate(model, X_test, y_test)})
    return pd.DataFrame(rows)
