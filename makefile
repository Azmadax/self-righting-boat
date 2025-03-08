.PHONY: help install-uv
.DEFAULT_GOAL := help

# Define the version of uv you want to install
UV_VERSION = 0.5.20

# Platform detection
ifeq ($(OS),Windows_NT)
	@echo "Windows platform detected"
	PLATFORM = windows
	INSTALL_CMD = curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		PLATFORM = linux
		INSTALL_CMD = wget -qO- https://astral.sh/uv/$(UV_VERSION)/install.sh | sh
	else ifeq ($(UNAME_S),Darwin)
		@echo "Darwin / Mac platform detected"
		PLATFORM = darwin
		INSTALL_CMD = curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh
	else
		$(error Unsupported platform: $(UNAME_S))
	endif
endif


help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-uv: ## Install uv project and packages manager
	@echo " ⏺ Checking uv installation..."
	@current_version=$$(if command -v uv >/dev/null 2>&1; then uv --version 2>/dev/null || echo "not_installed"; else echo "not_installed"; fi); \
	echo "UV version: $$current_version"; \
	echo "Expected version: $(UV_VERSION)"; \
	if [ "$$current_version" = "not_installed" ]; then \
		echo "uv is not installed. Installing version $(UV_VERSION)..."; \
		$(INSTALL_CMD); \
	elif echo "$$current_version" | grep -q "$(UV_VERSION)"; then \
		echo "uv is already installed with the correct version ($$current_version)."; \
	else \
		echo "Current version ($$current_version) does not match $(UV_VERSION). Updating..."; \
		$(INSTALL_CMD); \
	fi; \
	echo "uv installation/update complete."

demo: ## Launch demo scripts
	uv sync --package hydrostatic
	uv run examples/hydrostatic_2d_example.py

format: ## Format code consistently
	ruff format

lint: ## Clean code or warn user
	ruff format
	ruff check . --fix

test: ## Launch test
	uv sync --package hydrostatic
	uv run pytest packages/hydrostatic/tests

coverage: ## Launch coverage test
	coverage run -m pytest packages/hydrostatic/tests
	coverage html --omit="*/test*"






