#!/usr/bin/env python
"""Predicción de [Mg/Fe] a partir de otras abundancias y parámetros estelares."""
import argparse

from galah_ml import pipeline, plotting
from galah_ml.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ruta a un YAML de configuración alternativo")
    parser.add_argument("--sweeps", action="store_true", help="ejecuta también los barridos de hiperparámetros (lento)")
    args = parser.parse_args()
    plotting.set_plot_style()
    pipeline.run_mg_fe(load_config(args.config), sweeps=args.sweeps)


if __name__ == "__main__":
    main()
