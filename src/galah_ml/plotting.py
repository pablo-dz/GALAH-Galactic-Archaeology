"""Figuras del proyecto: outliers, diagrama de Toomre, barridos de hiperparámetros, matrices de confusión, etc."""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .preprocessing import iter_sigma_clipping

_LABEL_SIZE = 15


def set_plot_style() -> None:
    """Estilo de gráficos: usa ``scienceplots`` si está instalado (sin LaTeX) y paleta pastel de seaborn."""
    try:
        import scienceplots  # noqa: F401

        plt.style.use("science")
    except ImportError:
        pass
    plt.rcParams["text.usetex"] = False
    sns.set_style("ticks", {"legend.frameon": True})
    sns.set_palette("pastel", 8, desat=1)


def _save(fig, path: str | Path, dpi: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=_LABEL_SIZE)


# ---------------------------------------------------------------------------


def plot_outlier_histograms(
    df: pd.DataFrame, columns: list[str], path: str | Path, sigma: float = 5.0, dpi: int = 200
) -> None:
    """Histogramas de cada columna: distribución original, tras sigma clipping y percentiles 2.5 / 97.5."""
    per_row = 4
    n_rows = int(np.ceil(len(columns) / per_row))
    fig = plt.figure(figsize=(15, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, per_row, hspace=1, wspace=0.5)

    for j, (col, x, mask) in enumerate(iter_sigma_clipping(df, columns, sigma)):
        ax = fig.add_subplot(gs[j // per_row, j % per_row])
        bins = np.linspace(x.min(), x.max(), 20)
        ax.hist(x, bins=bins, alpha=0.7, label="Original")
        ax.hist(x[~mask], bins=bins, alpha=0.7, label="Sigma Clipped")
        ax.axvline(np.percentile(x, 2.5), color="red", lw=2, label="2.5 Percentile")
        ax.axvline(np.percentile(x, 97.5), color="red", lw=2, label="97.5 Percentile")
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("N", fontsize=12)
        ax.set_yscale("log")
        ax.legend(fontsize=5)
    _save(fig, path, dpi)


def plot_toomre(df: pd.DataFrame, radii: tuple[float, float, float], path: str | Path, dpi: int = 200) -> None:
    """Diagrama de Toomre con las tres componentes y las estrellas sin clasificar en gris."""
    r1, r2, r3 = radii
    theta = np.linspace(0, 2 * np.pi, 6000)
    v = df["V_UVW"]
    uw = np.sqrt(df["U_UVW"] ** 2 + df["W_UVW"] ** 2)
    none = (df[["thin_disk", "thick_disk", "halo"]] == 0).all(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(v[df["thin_disk"] == 1], uw[df["thin_disk"] == 1], s=0.5, alpha=0.25, label="Disco delgado")
    ax.scatter(v[none], uw[none], s=0.5, color="gray", alpha=0.01)
    ax.scatter(v[df["thick_disk"] == 1], uw[df["thick_disk"] == 1], s=0.5, alpha=0.25, label="Disco grueso")
    ax.scatter(v[df["halo"] == 1], uw[df["halo"] == 1], s=0.5, alpha=0.25, label="Halo")
    for r in radii:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="gray", linestyle="dashed")
        ax.text(0, r + 3, f"{r:g}")
    ax.set_ylim(0, 300)
    ax.set_xlim(-400, 200)
    ax.set_xlabel(r"$\mathrm{V_{UVW} \ (km \ s^{-1})}$", fontsize=_LABEL_SIZE)
    ax.set_ylabel(r"$\mathrm{\left( U^2_{UVW} + W^2_{UVW} \right)^{1/2} \ (km \ s^{-1})}$", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LABEL_SIZE, markerscale=20)
    _style_axes(ax)
    _save(fig, path, dpi)


def plot_metric_sweep(
    sweep: pd.DataFrame, param: str, metrics: dict[str, str], path: str | Path, dpi: int = 200
) -> None:
    """Curvas de métricas en función de un hiperparámetro. ``metrics`` mapea columna -> etiqueta."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for col, label in metrics.items():
        ax.plot(sweep[param], sweep[col], label=label)
    ax.set_xlabel(param, fontsize=_LABEL_SIZE)
    ax.set_ylabel("Metric Value", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LABEL_SIZE)
    _style_axes(ax)
    _save(fig, path, dpi)


def plot_confusion_matrix(cm: np.ndarray, labels, title: str, path: str | Path, dpi: int = 200) -> None:
    """Matriz de confusión normalizada."""
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    disp.ax_.set_xlabel("Clase Predicha")
    disp.ax_.set_ylabel("Clase Real")
    disp.ax_.set_title(title)
    _save(disp.figure_, path, dpi)


def plot_true_vs_pred(y_true, y_pred, title: str, path: str | Path, dpi: int = 200) -> None:
    """Densidad (hexbin) de valores predichos frente a valores reales."""
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap="magma", mincnt=1)
    cbar = fig.colorbar(hb, ax=ax, label="Density")
    cbar.ax.yaxis.label.set_fontsize(20)
    ax.set_xlabel("True Values", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Predicted Values", fontsize=_LABEL_SIZE)
    ax.set_title(title, fontsize=20)
    _style_axes(ax)
    _save(fig, path, dpi)


def plot_abundance_vs_age(
    df: pd.DataFrame, columns: list[str], path: str | Path, age_col: str = "age", dpi: int = 200
) -> None:
    """Diagramas de densidad [X/Fe] frente a la edad para las abundancias indicadas."""
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows + 2), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, columns):
        sns.histplot(x=df[age_col], y=df[col], bins=100, pthresh=0.1, cmap="viridis", cbar=False, ax=ax)
        element = col.replace("_fe", "/Fe").capitalize()
        ax.set_xlabel("Age (Gyr)")
        ax.set_ylabel(rf"[{element}] $\mathrm{{dex}}$")
        ax.set_ylim(-0.25, 0.50)
        ax.set_xlim(0, 12)
    for ax in axes[len(columns):]:
        fig.delaxes(ax)
    _save(fig, path, dpi)
