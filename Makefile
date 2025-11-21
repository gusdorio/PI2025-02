# ============================================================================
# Makefile for ML Operations Docker Environment
# ============================================================================
# This Makefile provides convenient commands for managing development and
# production Docker environments.
#
# Quick Start:
#   make help          - Show all available commands
#   make dev-up        - Start development environment
#   make prod-up       - Start production environment
# ============================================================================

.PHONY: help
.DEFAULT_GOAL := help

# Color output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
RESET  := \033[0m

# Docker Compose files
DEV_COMPOSE  := docker-compose.dev.yml
PROD_COMPOSE := docker-compose.prod.yml
VERSION_SCRIPT := ./scripts/version.sh


# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║  ML Operations Docker Environment - Available Commands   ║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)Development Commands:$(RESET)"
	@grep -E '^dev-[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Production Commands:$(RESET)"
	@grep -E '^prod-[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Utility Commands:$(RESET)"
	@grep -E '^[a-z][a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v -E '^(dev-|prod-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""


# ============================================================================
# DEVELOPMENT ENVIRONMENT
# ============================================================================

dev-build: ## Build development images (with cache)
	@echo "$(BLUE)Building development images with version tags...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/ml-model:$${VERSION_TAG}-dev \
		-t pi2025-02/ml-model:dev-latest \
		-t pi2025-02/ml-model:commit-$${GIT_COMMIT} \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/dashboard:$${VERSION_TAG}-dev \
		-t pi2025-02/dashboard:dev-latest \
		-t pi2025-02/dashboard:commit-$${GIT_COMMIT} \
		.

dev-build-nc: ## Build development images (no cache)
	@echo "$(BLUE)Building development images with version tags...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/ml-model:$${VERSION_TAG}-dev \
		-t pi2025-02/ml-model:dev-latest \
		-t pi2025-02/ml-model:commit-$${GIT_COMMIT} \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/dashboard:$${VERSION_TAG}-dev \
		-t pi2025-02/dashboard:dev-latest \
		-t pi2025-02/dashboard:commit-$${GIT_COMMIT} \
		--no-cache \
		.

dev-up: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) up

dev-up-build: ## Build and start development environment
	@echo "$(BLUE)Building development images with version tags...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
    docker build \
        --target ml-model \
        --build-arg VERSION_TAG=$${VERSION_TAG} \
        --build-arg GIT_COMMIT=$${GIT_COMMIT} \
        --build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
        --build-arg ENVIRONMENT=development \
        -t pi2025-02/ml-model:$${VERSION_TAG} \
        -t pi2025-02/ml-model:latest \
        -t pi2025-02/ml-model:commit-$${GIT_COMMIT} \
        .; \
    docker build \
        --target dashboard \
        --build-arg VERSION_TAG=$${VERSION_TAG} \
        --build-arg GIT_COMMIT=$${GIT_COMMIT} \
        --build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
        --build-arg ENVIRONMENT=development \
        -t pi2025-02/dashboard:$${VERSION_TAG} \
        -t pi2025-02/dashboard:latest \
		-t pi2025-02/dashboard:commit-$${GIT_COMMIT} \
        .; \
    docker-compose -f $(DEV_COMPOSE) up -d

dev-up-d: ## Start development environment (detached)
	@echo "$(GREEN)Starting development environment in background...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) up -d

dev-down: ## Stop development environment
	@echo "$(YELLOW)Stopping development environment...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) down

dev-down-v: ## Stop development environment and remove volumes
	@echo "$(YELLOW)Stopping development environment and removing volumes...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) down -v

dev-restart: ## Restart development environment
	@echo "$(YELLOW)Restarting development environment...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) restart

dev-logs: ## Show development logs (follow)
	docker-compose -f $(DEV_COMPOSE) logs -f

dev-logs-ml: ## Show ML model service logs
	docker-compose -f $(DEV_COMPOSE) logs -f ml-model

dev-logs-dashboard: ## Show dashboard service logs
	docker-compose -f $(DEV_COMPOSE) logs -f streamlit-dashboard

dev-status: ## Show development services status
	docker-compose -f $(DEV_COMPOSE) ps

dev-shell-ml: ## Open shell in ML model container
	docker-compose -f $(DEV_COMPOSE) exec ml-model /bin/bash

dev-shell-dashboard: ## Open shell in dashboard container
	docker-compose -f $(DEV_COMPOSE) exec streamlit-dashboard /bin/bash

dev-shell-db: ## Open MongoDB shell
	docker-compose -f $(DEV_COMPOSE) exec mongodb mongosh -u root -p root_password123 pi2502


# ============================================================================
# PRODUCTION ENVIRONMENT
# ============================================================================

prod-build: ## Build production images (with cache)
	@echo "$(BLUE)Building production images with version tags...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=production \
		-t pi2025-02/ml-model:$${VERSION_TAG} \
		-t pi2025-02/ml-model:latest \
		-t pi2025-02/ml-model:commit-$${GIT_COMMIT} \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=production \
		-t pi2025-02/dashboard:$${VERSION_TAG} \
		-t pi2025-02/dashboard:latest \
		-t pi2025-02/dashboard:commit-$${GIT_COMMIT} \
		.

prod-build-nc: ## Build production images (no cache)
	@echo "$(BLUE)Building production images with version tags...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=production \
		-t pi2025-02/ml-model:$${VERSION_TAG} \
		-t pi2025-02/ml-model:latest \
		-t pi2025-02/ml-model:commit-$${GIT_COMMIT} \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=production \
		-t pi2025-02/dashboard:$${VERSION_TAG} \
		-t pi2025-02/dashboard:latest \
		-t pi2025-02/dashboard:commit-$${GIT_COMMIT} \
		--no-cache \
		.

prod-up: ## Start production environment (detached)
	@echo "$(GREEN)Starting production environment...$(RESET)"
	docker-compose -f $(PROD_COMPOSE) up -d

prod-up-build: ## Build and start production environment
	@echo "$(BLUE)Building production images with version tags...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
    docker build \
        --target ml-model \
        --build-arg VERSION_TAG=$${VERSION_TAG} \
        --build-arg GIT_COMMIT=$${GIT_COMMIT} \
        --build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
        --build-arg ENVIRONMENT=production \
        -t pi2025-02/ml-model:$${VERSION_TAG} \
        -t pi2025-02/ml-model:latest \
        -t pi2025-02/ml-model:commit-$${GIT_COMMIT} \
        .; \
    docker build \
        --target dashboard \
        --build-arg VERSION_TAG=$${VERSION_TAG} \
        --build-arg GIT_COMMIT=$${GIT_COMMIT} \
        --build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
        --build-arg ENVIRONMENT=production \
        -t pi2025-02/dashboard:$${VERSION_TAG} \
        -t pi2025-02/dashboard:latest \
        -t pi2025-02/dashboard:commit-$${GIT_COMMIT} \
        .; \
    docker-compose -f $(PROD_COMPOSE) up -d

prod-down: ## Stop production environment
	@echo "$(YELLOW)Stopping production environment...$(RESET)"
	docker-compose -f $(PROD_COMPOSE) down

prod-down-v: ## Stop production environment and remove volumes
	@echo "$(YELLOW)Stopping production environment and removing volumes...$(RESET)"
	docker-compose -f $(PROD_COMPOSE) down -v

prod-restart: ## Restart production environment
	@echo "$(YELLOW)Restarting production environment...$(RESET)"
	docker-compose -f $(PROD_COMPOSE) restart

prod-logs: ## Show production logs (last 100 lines)
	docker-compose -f $(PROD_COMPOSE) logs --tail=100

prod-logs-f: ## Show production logs (follow)
	docker-compose -f $(PROD_COMPOSE) logs -f

prod-logs-ml: ## Show ML model service logs
	docker-compose -f $(PROD_COMPOSE) logs --tail=100 ml-model

prod-logs-dashboard: ## Show dashboard service logs
	docker-compose -f $(PROD_COMPOSE) logs --tail=100 streamlit-dashboard

prod-status: ## Show production services status
	docker-compose -f $(PROD_COMPOSE) ps

prod-shell-ml: ## Open shell in ML model container
	docker-compose -f $(PROD_COMPOSE) exec ml-model /bin/bash

prod-shell-dashboard: ## Open shell in dashboard container
	docker-compose -f $(PROD_COMPOSE) exec streamlit-dashboard /bin/bash

prod-shell-db: ## Open MongoDB shell
	docker-compose -f $(PROD_COMPOSE) exec mongodb mongosh -u root -p root_password123 pi2502


# ============================================================================
# UTILITY COMMANDS
# ============================================================================

clean: ## Remove all stopped containers and unused images
	@echo "$(YELLOW)Cleaning up Docker resources...$(RESET)"
	docker system prune -f

clean-all: ## Remove all containers, images, and volumes (DESTRUCTIVE)
	@echo "$(YELLOW)WARNING: This will remove all Docker resources!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose -f $(DEV_COMPOSE) down -v --remove-orphans; \
		docker-compose -f $(PROD_COMPOSE) down -v --remove-orphans; \
		docker system prune -a --volumes -f; \
	fi

validate-dev: ## Validate development compose file
	@echo "$(BLUE)Validating development compose file...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) config --quiet && echo "$(GREEN)✓ Valid$(RESET)" || echo "$(YELLOW)✗ Invalid$(RESET)"

validate-prod: ## Validate production compose file
	@echo "$(BLUE)Validating production compose file...$(RESET)"
	docker-compose -f $(PROD_COMPOSE) config --quiet && echo "$(GREEN)✓ Valid$(RESET)" || echo "$(YELLOW)✗ Invalid$(RESET)"

validate-all: validate-dev validate-prod ## Validate all compose files

images: ## List all project images
	@echo "$(BLUE)Project images:$(RESET)"
	@docker images | grep -E "pi2025-02" || echo "No images found"

volumes: ## List all project volumes
	@echo "$(BLUE)Project volumes:$(RESET)"
	@docker volume ls | grep -E "PI2502" || echo "No volumes found"

networks: ## List all project networks
	@echo "$(BLUE)Project networks:$(RESET)"
	@docker network ls | grep -E "PI2502" || echo "No networks found"

ps: ## Show all running containers (dev and prod)
	@echo "$(BLUE)All running containers:$(RESET)"
	@docker ps --filter "name=ml-model" --filter "name=streamlit-dashboard" --filter "name=mongodb"
