-include .local.mk

ROOT_DIR = sakhilabs
VENV = venv_sakhi
PIP = pip
PYTHON ?= python

virtual-env:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install uv pip-tools ruff isort
	. $(VENV)/bin/activate && uv $(PIP) install -r requirements.txt

compile-requirements:
	. $(VENV)/bin/activate && pip-compile requirements.in
	
clean-pycache:
	find $(ROOT_DIR) -type d -name "__pycache__" -exec rm -rf {} +

clean: clean-pycache
	rm -rf $(VENV)

reset: clean virtual-env

lint:
	. $(VENV)/bin/activate && ruff format $(ROOT_DIR)
	. $(VENV)/bin/activate && isort $(ROOT_DIR)
