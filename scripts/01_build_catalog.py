#!/usr/bin/env python
"""Construye data/processed/ready_data.csv a partir de los catálogos FITS de GALAH DR4."""
import argparse

from galah_ml import pipeline
from galah_ml.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ruta a un YAML de configuración alternativo")
    args = parser.parse_args()
    pipeline.build_catalog(load_config(args.config))


if __name__ == "__main__":
    main()
