.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo [*] Updating git submodule recursively and remotely
	@git pull --recurse-submodules
	@git submodule update --init --recursive
	@echo [*] Creating virtual environment and installing dependencies using poetry
	@poetry lock
	@poetry install --with dev
	@poetry run pre-commit install

.PHONY: check
check: ## Run code quality tools
	@echo [*] Checking Poetry lock file consistency with 'pyproject.toml': Running poetry check --lock
	@poetry check --lock
	@echo [*] Linting code: Running pre-commit
	@poetry run pre-commit run -a

.PHONY: test
test: ## Run unit tests
	@echo [*] Running Unit Tests
	@poetry run pytest tests/

.PHONY: clean
clean: ## Remove generated files
	@rm -fr .mypy_cache .ruff_cache .pytest_cache .vscode
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find src -regex '^.*\(__pycache__\|\.py[co]\)$$' -delete
