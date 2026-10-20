# Local development workflow. Requires: Python >= 3.10, Node >= 20.9 (see .nvmrc), npm, make.
# On Windows use WSL, or run the underlying commands shown here by hand.

VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
BIN     := $(abspath $(VENV)/bin)
API_DIR := services/api
WEB_DIR := apps/web
ML_DIR  := ml

.DEFAULT_GOAL := help
.PHONY: help setup setup-py setup-web api web dev validate-data train report test test-api test-ml test-web \
        lint lint-py lint-web typecheck typecheck-py typecheck-web format check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

# -- setup -------------------------------------------------------------------

setup: setup-py setup-web ## Install all dependencies (Python venv + npm)

setup-py: ## Create .venv and install API + ML packages (pinned by constraints.txt)
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -c constraints.txt -e "$(API_DIR)[dev]" -e "$(ML_DIR)[dev]"

setup-web: ## Install frontend dependencies from package-lock.json
	cd $(WEB_DIR) && npm ci

# -- run ---------------------------------------------------------------------

api: ## Run the FastAPI backend on http://localhost:8000 (auto-reload)
	cd $(API_DIR) && $(BIN)/uvicorn app.main:app --reload --port 8000

web: ## Run the Next.js frontend on http://localhost:3000
	cd $(WEB_DIR) && npm run dev

dev: ## Run backend and frontend together (Ctrl+C stops both)
	$(MAKE) -j2 api web

validate-data: ## Validate Fake.csv + True.csv (default: ml/data); DATA_DIR=... to override
	$(PYTHON) -m fakenews_ml.validate --data-dir "$(or $(DATA_DIR),$(ML_DIR)/data)"

train: ## Train + compare + evaluate; writes a versioned model (see ml/README.md)
ifdef DATA
	$(PYTHON) -m fakenews_ml.train --data "$(DATA)"
else ifdef FAKE
	$(PYTHON) -m fakenews_ml.train --fake-csv "$(FAKE)" --real-csv "$(REAL)"
else
	$(PYTHON) -m fakenews_ml.train --data-dir "$(or $(DATA_DIR),$(ML_DIR)/data)"
endif

report: ## Print the evaluation recorded in the active model's metadata
	$(PYTHON) -m fakenews_ml.report

# -- quality -----------------------------------------------------------------

test-api: ## Backend tests
	cd $(API_DIR) && $(BIN)/pytest

test-ml: ## ML training-code tests
	cd $(ML_DIR) && $(BIN)/pytest

test-web: ## Frontend tests
	cd $(WEB_DIR) && npm test

test: test-api test-ml test-web ## All tests

lint-py: ## Ruff lint + format check
	cd $(API_DIR) && $(BIN)/ruff check . && $(BIN)/ruff format --check .
	cd $(ML_DIR) && $(BIN)/ruff check . && $(BIN)/ruff format --check .

lint-web: ## ESLint + Prettier check
	cd $(WEB_DIR) && npm run lint && npm run format:check

lint: lint-py lint-web ## All linters

typecheck-py: ## mypy (strict)
	cd $(API_DIR) && $(BIN)/mypy
	cd $(ML_DIR) && $(BIN)/mypy

typecheck-web: ## tsc --noEmit
	cd $(WEB_DIR) && npm run typecheck

typecheck: typecheck-py typecheck-web ## All type checks

format: ## Auto-format Python and TypeScript
	cd $(API_DIR) && $(BIN)/ruff check --fix . && $(BIN)/ruff format .
	cd $(ML_DIR) && $(BIN)/ruff check --fix . && $(BIN)/ruff format .
	cd $(WEB_DIR) && npm run format

check: lint typecheck test ## Everything CI runs (except the production build)
	cd $(WEB_DIR) && npm run build

clean: ## Remove caches and build output (keeps .venv, node_modules and artifacts)
	rm -rf $(WEB_DIR)/.next
	find . -name __pycache__ -not -path "./$(VENV)/*" -not -path "*/node_modules/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache $(API_DIR)/.pytest_cache $(API_DIR)/.mypy_cache \
	       $(API_DIR)/.ruff_cache $(ML_DIR)/.pytest_cache $(ML_DIR)/.mypy_cache $(ML_DIR)/.ruff_cache
