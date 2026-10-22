PY ?= .venv/bin/python

.PHONY: setup all test data

setup:  ## create the virtual environment
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

all:    ## reproduce every table and figure in reports/ from the cached data
	$(PY) run_all.py

test:
	$(PY) -m pytest -q

data:   ## re-download EDGAR, Treasury.gov and FRED data, then rebuild (needs EDGAR_USER_AGENT)
	$(PY) run_all.py --refresh
