.PHONY: all install run test clean

VENV := .venv
ifeq ($(OS),Windows_NT)
PYTHON := python
VENV_PY := $(VENV)/Scripts/python.exe
VENV_PIP := $(VENV)/Scripts/pip.exe
else
PYTHON := python3
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
endif

all: install run

$(VENV_PY):
	$(PYTHON) -m venv $(VENV)

install: $(VENV_PY)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

run: install
	$(VENV_PY) run.py

test: install
	$(VENV_PY) -m unittest discover -s tests -v

clean:
	rm -rf $(VENV) __pycache__ src/**/__pycache__ tests/__pycache__
