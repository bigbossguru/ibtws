.PHONY: help install update check lint format test clean

.DEFAULT_GOAL := help

help:  ## Show this help messagee
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the poetry environment and pre-commit hooks
	@echo "📦 Initialising git submodules"
	@git submodule update --init --recursive
	@echo "🐍 Creating virtual environment and installing dependencies using poetry"
	@poetry install --with dev
	@poetry run pre-commit install

update:  ## Pull latest code and update submodules
	@echo "⬇️  Pulling latest code and submodules"
	@git pull --recurse-submodules
	@git submodule update --init --recursive

check:  ## Run all code quality tools (lock check + pre-commit)
	@echo "🔒 Checking poetry lock file consistency with pyproject.toml"
	@poetry check --lock
	@echo "🧹 Linting code: running pre-commit on all files"
	@poetry run pre-commit run -a

lint:  ## Run ruff linter
	@poetry run ruff check src tests

format:  ## Auto-format code with ruff
	@poetry run ruff format src tests
	@poetry run ruff check --fix src tests

test:  ## Run unit tests (pass extra args via ARGS=...)
	@echo "🧪 Running unit tests"
	@poetry run pytest tests/ $(ARGS)

clean:  ## Remove generated caches and build artifacts
	@rm -rf .mypy_cache .ruff_cache .pytest_cache .vscode .coverage htmlcov dist build
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
