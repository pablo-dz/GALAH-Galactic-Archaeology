.PHONY: install catalog classify mg age all sweeps test

install:
	pip install -e ".[dev]"

catalog:
	python scripts/01_build_catalog.py

classify:
	python scripts/02_classify_components.py

mg:
	python scripts/03_predict_mg_fe.py

age:
	python scripts/04_predict_age.py

all: catalog classify mg age

sweeps:
	python scripts/02_classify_components.py --sweeps
	python scripts/03_predict_mg_fe.py --sweeps
	python scripts/04_predict_age.py --sweeps

test:
	pytest
