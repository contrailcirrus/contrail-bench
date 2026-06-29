# ==============================================================================
# Contrail Bench – automated tasks
# ==============================================================================

.DEFAULT_GOAL  := help
PYTHON         := python
# PIP  		   := python -m pip
PIP  		      := uv pip # use uv: https://docs.astral.sh/uv/
PACKAGE        := contrailscope
PYTEST         := pytest
RUFF           := ruff
PRECOMMIT      := pre-commit
SPHINX         := sphinx-build
DOCS_DIR       := docs
DOCS_BUILD_DIR := docs/_build
DIST_DIR       := dist

SHELL := /bin/bash  # override default /bin/sh
TAG ?= $(shell git describe --tags)

# Colors
COLOR_YELLOW = \033[0;33m
END_COLOR = \033[0m

.PHONY: help

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
help:
	@echo ""
	@echo "Contrail Bench tasks"
	@echo "===================="
	@echo ""
	@echo "Setup:"
	@echo "  install          	Install all local dependencies"
	@echo "  pre-commit-install	Install git pre-commit hooks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build HTML documentation"
	@echo "  docs-serve       Build and serve docs on http://localhost:8000"
	@echo "  docs-clean       Remove docs build artefacts"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            Remove all build/test/cache artifacts"
	@echo ""


# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
venv:
	# requires uv: https://docs.astral.sh/uv/
	uv venv --seed
	@echo "Activate virtual environment in your shell:"
	@echo ""
	@echo "# macOS/Linux"
	@echo "source .venv/bin/activate"
	@echo ""
	@echo "# Windows (PowerShell)"
	@echo ".venv\Scripts\Activate.ps1"
	@echo ""
	@echo "# Windows (CMD)"
	@echo ".venv\Scripts\activate.bat"

install:
	$(PIP) install -e ".[complete]"

	# install pre-commit
	$(PRECOMMIT) install

pre-commit-install:
	$(PRECOMMIT) install

clean: docs-clean
	rm -rf $(DIST_DIR) build/ .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

remove: clean
	$(PIP) uninstall contrailbench

# ------------------------------------------------------------------------------
# QC, Test
# ------------------------------------------------------------------------------

lint:
	$(RUFF) check contrailbench docs
	$(RUFF) format --check contrailbench docs

format:
	$(RUFF) check --fix contrailbench docs
	$(RUFF) format contrailbench docs

pre-commit:
	$(PRECOMMIT) run --all-files

# https://taplo.tamasfe.dev/
taplo:
	taplo format pyproject.toml --option indent_string='    '

# https://yamllint.readthedocs.io/en/stable/configuration.html
yamllint:
	yamllint -d "{extends: default, rules: {line-length: {max: 100}}}" .

mypy:
	mypy contrailbench

	# switch over mid-2026
	# ty check contrailscope

pytest:
	($PYTEST)

pytest-cov:
	($PYTEST) \
		-v \
		--cov=contrailscope \
		--cov-report=html:coverage \
		--cov-report=term-missing \
		--durations=10

test: lint mypy pytest doctest nb-test


# -----------
# Release
# -----------


# ------------------------------------------------------------------------------
# Documentation
# ------------------------------------------------------------------------------
doc8:
	doc8 docs

docs: doc8
	$(SPHINX) -b html -W --keep-going $(DOCS_DIR) $(DOCS_BUILD_DIR)/html

docs-serve: docs
	sphinx-autobuild \
		--re-ignore _build\/.* \
		-b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html

docs-clean:
	rm -rf $(DOCS_BUILD_DIR)

doctest:
	($PYTEST) --doctest-modules \
		contrailscope -vv

docs-pdf: doc8
	$(SPHINX) -b latex $(DOCS_DIR) $(DOCS_BUILD_DIR)/latex

	# this needs to get run twice
	cd $(DOCS_BUILD_DIR)/latex && make
	cd $(DOCS_BUILD_DIR)/latex && make

	echo "PDF exported to $(DOCS_BUILD_DIR)/latex/contrailbench.pdf"

nb-format-check:
	$(RUFF) format --check docs/**/*.ipynb

# Note must be kept in sync with 
# `.pre-commit-config.yaml` and `make nb-clean-check`
nb-clean:
	nb-clean clean docs/**/*.ipynb \
        --remove-empty-cells \
		--preserve-cell-metadata tags \
		--preserve-cell-outputs \
		--preserve-execution-counts

nb-clean-check:
	nb-clean check docs/**/*.ipynb \
        --remove-empty-cells \
		--preserve-cell-metadata tags \
		--preserve-cell-outputs \
		--preserve-execution-counts

# Check for broken links in notebooks
# https://github.com/jupyterlab/pytest-check-links
nb-check-links:
	$(PYTEST) --check-links
		docs/notebooks/*.ipynb

# Makes sure that all cells can execute without errors
# Add `nbval-skip` cell tag if you want to skip a cell
# Add `nbval-check-output` cell tag if you want to specifically compare cell output
nb-test: nb-clean-check nb-format-check nb-check-links
	$(PYTEST) --nbval-lax \
		docs/notebooks

# Execute all notebooks in docs
# Add `skip-execution` cell tag if you want to skip a cell
# Add `raises-exception` cell tag if you know the cell raises exception
nb-execute: nb-format-check nb-check-links
	$(JUPYTER) nbconvert --inplace \
		--to notebook \
		--execute \
		docs/notebooks/*.ipynb

	# clean notebooks after execution
	make nb-clean
