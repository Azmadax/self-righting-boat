.PHONY: help install-micromamba build-env setup-env install-poetry install-uv init-uv sync-dependencies
.DEFAULT_GOAL := help

PROJECT = self-righting-boat
MICROMAMBA_BIN = $(HOME)/.local/bin/micromamba
PATH_ADDITION = export PATH=$(HOME)/.local/bin:$$PATH;

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-micromamba: ## Install micromamba (conda replacement) non-interactively and add to PATH
	@if [ ! -x "$(MICROMAMBA_BIN)" ]; then \
		echo "Micromamba not found, installing..."; \
		curl -Ls https://micro.mamba.pm/install.sh | bash -s -- -b -f; \
	else \
		echo "Micromamba is already installed."; \
	fi
	@echo "Adding Micromamba to PATH for subsequent commands."

build-env: install-micromamba ## Setup environment (only when updating python or poetry version)
	@$(PATH_ADDITION) bash -c 'eval "$$($(MICROMAMBA_BIN) shell hook --shell bash)" && \
		$(MICROMAMBA_BIN) create -p /tmp/bootstrap -c conda-forge conda-lock -y && \
		micromamba activate /tmp/bootstrap && \
		conda-lock -f environment.yml'

setup-env: install-micromamba ## Create copy of environment
	@$(PATH_ADDITION) bash -c 'eval "$$($(MICROMAMBA_BIN) shell hook --shell bash)" && \
	if $(MICROMAMBA_BIN) env list | grep -q $(PROJECT); then \
		$(MICROMAMBA_BIN) env remove --name $(PROJECT); \
	else \
		echo "Environment $(PROJECT) does not exist, creating."; \
	fi && \
	micromamba activate /tmp/bootstrap && \
	conda-lock install --name $(PROJECT) conda-lock.yml'

install-poetry: ## Be sure to activate env
	poetry install --no-root


