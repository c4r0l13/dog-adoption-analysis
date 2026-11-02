.PHONY: install data run test lint

install:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

run:              ## analysis + model + figures -> reports/ (~30 s)
	python -m adopt.pipeline

test:
	pytest -q

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts
