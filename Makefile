VENV := .venv
PYTHON := $(VENV)/bin/python
PRE_COMMIT := $(VENV)/bin/pre-commit


.PHONY: help setup dev venv-create venv-sync venv-diff venv-lock fmt lint test run clean activate visualize

help: ## Show available commands
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

activate: ## Show how to activate venv
	@echo "Run: source $(VENV)/bin/activate"

dev: ## Full dev setup: create venv, install from lock, install pre-commit hooks
	make venv-create
	make venv-sync
	$(PRE_COMMIT) clean
	$(PRE_COMMIT) install

fmt: ## Format code with black and ruff
	$(VENV)/bin/black .
	$(VENV)/bin/ruff check . --fix

lint: ## Run static analysis
	@$(VENV)/bin/ruff check .
	@$(VENV)/bin/mypy .

test: ## Run tests
	$(VENV)/bin/pytest tests

run: ## Run main application
	$(PYTHON) src/your_project_name/main.py

venv-create: ## Create .venv if it doesn't exist
	uv self update
	@test -d $(VENV) || uv venv $(VENV)

venv-lock: ## Recompile uv.lock from pyproject.toml
	uv lock

venv-sync: ## Install exact versions from uv.lock for dev group
	uv sync --group dev

venv-diff: ## Detect if uv.lock is out of sync with pyproject.toml (ignoring comments)
	uv sync --dry-run

venv-clean: ## Remove virtualenv
	rm -rf $(VENV)

visualize:
	PYTHONPATH=. python scripts/visualize_test.py
