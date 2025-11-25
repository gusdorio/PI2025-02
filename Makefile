# ============================================================================
# Makefile for ML Operations Docker Environment
# ============================================================================
# This Makefile provides convenient commands for managing development,
# staging, and Azure deployment environments.
#
# Quick Start:
#   make help          - Show all available commands
#   make dev-up        - Start development environment (local MongoDB)
#   make staging-up    - Start staging environment (CosmosDB)
#   make azure-deploy  - Deploy to Azure Container Apps
# ============================================================================

.PHONY: help
.DEFAULT_GOAL := help

# Color output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
RESET  := \033[0m

# Docker Compose files
DEV_COMPOSE     := docker-compose.dev.yml
STAGING_COMPOSE := docker-compose.staging.yml
VERSION_SCRIPT  := ./scripts/version.sh


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
	@grep -hE '^dev-[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Staging Commands:$(RESET)"
	@grep -hE '^staging-[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Azure Commands:$(RESET)"
	@grep -hE '^azure-[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Utility Commands:$(RESET)"
	@grep -hE '^[a-z][a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE '^(dev-|staging-|azure-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""


# ============================================================================
# DEVELOPMENT ENVIRONMENT
# ============================================================================

dev-help: ## Show development environment commands
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║         Development Environment Commands                 ║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)Build:$(RESET)"
	@echo "  $(GREEN)make dev-build$(RESET)              - Build images (with cache)"
	@echo "  $(GREEN)make dev-build-nc$(RESET)           - Build images (no cache)"
	@echo ""
	@echo "$(YELLOW)Run:$(RESET)"
	@echo "  $(GREEN)make dev-up$(RESET)                 - Start environment (attached)"
	@echo "  $(GREEN)make dev-up-d$(RESET)               - Start environment (detached)"
	@echo "  $(GREEN)make dev-up-build$(RESET)           - Build and start (detached)"
	@echo "  $(GREEN)make dev-down$(RESET)               - Stop environment"
	@echo "  $(GREEN)make dev-down-v$(RESET)             - Stop and remove volumes"
	@echo "  $(GREEN)make dev-restart$(RESET)            - Restart environment"
	@echo ""
	@echo "$(YELLOW)Logs & Status:$(RESET)"
	@echo "  $(GREEN)make dev-status$(RESET)             - Show services status"
	@echo "  $(GREEN)make dev-logs$(RESET)               - Show logs (follow)"
	@echo "  $(GREEN)make dev-logs-ml$(RESET)            - Show ml-model logs"
	@echo "  $(GREEN)make dev-logs-dashboard$(RESET)     - Show dashboard logs"
	@echo ""
	@echo "$(YELLOW)Shell Access:$(RESET)"
	@echo "  $(GREEN)make dev-shell-ml$(RESET)           - Shell into ml-model container"
	@echo "  $(GREEN)make dev-shell-dashboard$(RESET)    - Shell into dashboard container"
	@echo "  $(GREEN)make dev-shell-db$(RESET)           - Open MongoDB shell"
	@echo ""

dev-build: ## Build development images (with cache)
	@echo "$(BLUE)Building development images...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/ml-model:dev \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/dashboard:dev \
		.

dev-build-nc: ## Build development images (no cache)
	@echo "$(BLUE)Building development images (no cache)...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/ml-model:dev \
		--no-cache \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/dashboard:dev \
		--no-cache \
		.

dev-up: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) up

dev-up-build: ## Build and start development environment
	@echo "$(BLUE)Building and starting development environment...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/ml-model:dev \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=development \
		-t pi2025-02/dashboard:dev \
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
# STAGING ENVIRONMENT (uses CosmosDB, mirrors Azure)
# ============================================================================

staging-help: ## Show staging environment commands
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║         Staging Environment Commands                     ║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "  Staging uses Azure CosmosDB for pre-deployment validation."
	@echo "  Requires COSMOSDB_* credentials in .env file."
	@echo ""
	@echo "$(YELLOW)Build:$(RESET)"
	@echo "  $(GREEN)make staging-build$(RESET)          - Build images (with cache)"
	@echo "  $(GREEN)make staging-build-nc$(RESET)       - Build images (no cache)"
	@echo ""
	@echo "$(YELLOW)Run:$(RESET)"
	@echo "  $(GREEN)make staging-up$(RESET)             - Start environment (detached)"
	@echo "  $(GREEN)make staging-up-build$(RESET)       - Build and start (detached)"
	@echo "  $(GREEN)make staging-down$(RESET)           - Stop environment"
	@echo "  $(GREEN)make staging-down-v$(RESET)         - Stop and remove volumes"
	@echo "  $(GREEN)make staging-restart$(RESET)        - Restart environment"
	@echo ""
	@echo "$(YELLOW)Logs & Status:$(RESET)"
	@echo "  $(GREEN)make staging-status$(RESET)         - Show services status"
	@echo "  $(GREEN)make staging-logs$(RESET)           - Show logs (last 100 lines)"
	@echo "  $(GREEN)make staging-logs-f$(RESET)         - Show logs (follow)"
	@echo "  $(GREEN)make staging-logs-ml$(RESET)        - Show ml-model logs"
	@echo "  $(GREEN)make staging-logs-dashboard$(RESET) - Show dashboard logs"
	@echo ""
	@echo "$(YELLOW)Shell Access:$(RESET)"
	@echo "  $(GREEN)make staging-shell-ml$(RESET)       - Shell into ml-model container"
	@echo "  $(GREEN)make staging-shell-dashboard$(RESET)- Shell into dashboard container"
	@echo ""

staging-build: ## Build staging images (with cache)
	@echo "$(BLUE)Building staging images...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=staging \
		-t pi2025-02/ml-model:staging \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=staging \
		-t pi2025-02/dashboard:staging \
		.

staging-build-nc: ## Build staging images (no cache)
	@echo "$(BLUE)Building staging images (no cache)...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=staging \
		-t pi2025-02/ml-model:staging \
		--no-cache \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=staging \
		-t pi2025-02/dashboard:staging \
		--no-cache \
		.

staging-up: ## Start staging environment (detached)
	@echo "$(GREEN)Starting staging environment...$(RESET)"
	docker-compose -f $(STAGING_COMPOSE) up -d

staging-up-build: ## Build and start staging environment
	@echo "$(BLUE)Building and starting staging environment...$(RESET)"
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=staging \
		-t pi2025-02/ml-model:staging \
		.; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=staging \
		-t pi2025-02/dashboard:staging \
		.; \
	docker-compose -f $(STAGING_COMPOSE) up -d

staging-down: ## Stop staging environment
	@echo "$(YELLOW)Stopping staging environment...$(RESET)"
	docker-compose -f $(STAGING_COMPOSE) down

staging-down-v: ## Stop staging environment and remove volumes
	@echo "$(YELLOW)Stopping staging environment and removing volumes...$(RESET)"
	docker-compose -f $(STAGING_COMPOSE) down -v

staging-restart: ## Restart staging environment
	@echo "$(YELLOW)Restarting staging environment...$(RESET)"
	docker-compose -f $(STAGING_COMPOSE) restart

staging-logs: ## Show staging logs (last 100 lines)
	docker-compose -f $(STAGING_COMPOSE) logs --tail=100

staging-logs-f: ## Show staging logs (follow)
	docker-compose -f $(STAGING_COMPOSE) logs -f

staging-logs-ml: ## Show ML model service logs
	docker-compose -f $(STAGING_COMPOSE) logs --tail=100 ml-model

staging-logs-dashboard: ## Show dashboard service logs
	docker-compose -f $(STAGING_COMPOSE) logs --tail=100 streamlit-dashboard

staging-status: ## Show staging services status
	docker-compose -f $(STAGING_COMPOSE) ps

staging-shell-ml: ## Open shell in ML model container
	docker-compose -f $(STAGING_COMPOSE) exec ml-model /bin/bash

staging-shell-dashboard: ## Open shell in dashboard container
	docker-compose -f $(STAGING_COMPOSE) exec streamlit-dashboard /bin/bash


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
		docker-compose -f $(STAGING_COMPOSE) down -v --remove-orphans; \
		docker system prune -a --volumes -f; \
	fi

validate-dev: ## Validate development compose file
	@echo "$(BLUE)Validating development compose file...$(RESET)"
	docker-compose -f $(DEV_COMPOSE) config --quiet && echo "$(GREEN)✓ Valid$(RESET)" || echo "$(YELLOW)✗ Invalid$(RESET)"

validate-staging: ## Validate staging compose file
	@echo "$(BLUE)Validating staging compose file...$(RESET)"
	docker-compose -f $(STAGING_COMPOSE) config --quiet && echo "$(GREEN)✓ Valid$(RESET)" || echo "$(YELLOW)✗ Invalid$(RESET)"

validate-all: validate-dev validate-staging ## Validate all compose files

images: ## List all project images
	@echo "$(BLUE)Project images:$(RESET)"
	@docker images | grep -E "pi2025-02" || echo "No images found"

volumes: ## List all project volumes
	@echo "$(BLUE)Project volumes:$(RESET)"
	@docker volume ls | grep -E "PI2502" || echo "No volumes found"

networks: ## List all project networks
	@echo "$(BLUE)Project networks:$(RESET)"
	@docker network ls | grep -E "PI2502" || echo "No networks found"

ps: ## Show all running containers (dev and staging)
	@echo "$(BLUE)All running containers:$(RESET)"
	@docker ps --filter "name=ml-model" --filter "name=streamlit-dashboard" --filter "name=mongodb"


# ============================================================================
# AZURE DEPLOYMENT COMMANDS
# ============================================================================
# These commands manage deployment to Azure Container Apps.
# For initial setup (CosmosDB, Environment), see azure-cli-commands-reference.md
#
# Quick Start:
#   make azure-check    - Validate prerequisites
#   make azure-deploy   - Full deployment (build + push + update)
#   make azure-status   - Check deployment status
# ============================================================================

# Load .env file if it exists
-include .env
export

# Azure scripts
AZURE_PREFLIGHT := ./scripts/azure-preflight.sh

azure-help: ## Show Azure deployment commands
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║         Azure Deployment Commands                        ║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)Validation:$(RESET)"
	@echo "  $(GREEN)make azure-check$(RESET)            - Run all pre-flight checks"
	@echo ""
	@echo "$(YELLOW)Build & Push:$(RESET)"
	@echo "  $(GREEN)make azure-login$(RESET)            - Login to Azure Container Registry"
	@echo "  $(GREEN)make azure-build$(RESET)            - Build images with ACR tags"
	@echo "  $(GREEN)make azure-push$(RESET)             - Push images to ACR"
	@echo ""
	@echo "$(YELLOW)Deploy & Update:$(RESET)"
	@echo "  $(GREEN)make azure-deploy$(RESET)           - Full deploy (build + push + update)"
	@echo "  $(GREEN)make azure-update-ml$(RESET)        - Update ml-model container app"
	@echo "  $(GREEN)make azure-update-dashboard$(RESET) - Update dashboard container app"
	@echo "  $(GREEN)make azure-sync-env$(RESET)         - Sync env vars from .env to apps"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(RESET)"
	@echo "  $(GREEN)make azure-status$(RESET)           - Show container apps status"
	@echo "  $(GREEN)make azure-urls$(RESET)             - Show application URLs"
	@echo "  $(GREEN)make azure-env-vars$(RESET)         - Show env vars from both apps"
	@echo "  $(GREEN)make azure-logs-ml$(RESET)          - Stream ml-model logs"
	@echo "  $(GREEN)make azure-logs-dashboard$(RESET)   - Stream dashboard logs"
	@echo ""

azure-check: ## Run Azure pre-flight validation checks
	@$(AZURE_PREFLIGHT) all

azure-login: ## Login to Azure Container Registry
	@echo "$(BLUE)Logging in to Azure Container Registry...$(RESET)"
	@$(AZURE_PREFLIGHT) env || exit 1
	@az acr login --name $(ACR_NAME)
	@echo "$(GREEN)Successfully logged in to $(ACR_NAME)$(RESET)"

azure-build: ## Build images for Azure Container Registry
	@echo "$(BLUE)Building images for Azure Container Registry...$(RESET)"
	@$(AZURE_PREFLIGHT) env || exit 1
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	echo "$(BLUE)Version: $${VERSION_TAG}$(RESET)"; \
	echo ""; \
	echo "$(BLUE)Building ml-model...$(RESET)"; \
	docker build \
		--target ml-model \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=production \
		-t $(ACR_LOGIN_SERVER)/pi2025-02/ml-model:$${VERSION_TAG} \
		-t $(ACR_LOGIN_SERVER)/pi2025-02/ml-model:latest \
		. && \
	echo ""; \
	echo "$(BLUE)Building dashboard...$(RESET)"; \
	docker build \
		--target dashboard \
		--build-arg VERSION_TAG=$${VERSION_TAG} \
		--build-arg GIT_COMMIT=$${GIT_COMMIT} \
		--build-arg BUILD_TIMESTAMP=$${BUILD_TIMESTAMP} \
		--build-arg ENVIRONMENT=production \
		-t $(ACR_LOGIN_SERVER)/pi2025-02/dashboard:$${VERSION_TAG} \
		-t $(ACR_LOGIN_SERVER)/pi2025-02/dashboard:latest \
		. && \
	echo ""; \
	echo "$(GREEN)Images built:$(RESET)"; \
	echo "  $(ACR_LOGIN_SERVER)/pi2025-02/ml-model:$${VERSION_TAG}"; \
	echo "  $(ACR_LOGIN_SERVER)/pi2025-02/dashboard:$${VERSION_TAG}"

azure-push: ## Push images to Azure Container Registry
	@echo "$(BLUE)Pushing images to ACR...$(RESET)"
	@$(AZURE_PREFLIGHT) env || exit 1
	@export $$($(VERSION_SCRIPT) | grep -v "^$$"); \
	echo "Pushing ml-model..."; \
	docker push $(ACR_LOGIN_SERVER)/pi2025-02/ml-model:$${VERSION_TAG} && \
	docker push $(ACR_LOGIN_SERVER)/pi2025-02/ml-model:latest && \
	echo "Pushing dashboard..."; \
	docker push $(ACR_LOGIN_SERVER)/pi2025-02/dashboard:$${VERSION_TAG} && \
	docker push $(ACR_LOGIN_SERVER)/pi2025-02/dashboard:latest && \
	echo "$(GREEN)Images pushed successfully$(RESET)"

azure-update-ml: ## Update ml-model container app with latest image
	@echo "$(BLUE)Updating ml-model container app...$(RESET)"
	@$(AZURE_PREFLIGHT) env || exit 1
	@az containerapp update \
		--name ml-model \
		--resource-group $(RESOURCE_GROUP) \
		--image "$(ACR_LOGIN_SERVER)/pi2025-02/ml-model:latest"
	@echo "$(GREEN)ml-model updated$(RESET)"

azure-update-dashboard: ## Update dashboard container app with latest image
	@echo "$(BLUE)Updating dashboard container app...$(RESET)"
	@$(AZURE_PREFLIGHT) env || exit 1
	@az containerapp update \
		--name dashboard \
		--resource-group $(RESOURCE_GROUP) \
		--image "$(ACR_LOGIN_SERVER)/pi2025-02/dashboard:latest"
	@echo "$(GREEN)dashboard updated$(RESET)"

azure-deploy: azure-login azure-build azure-push ## Full deploy: build, push, and update both apps
	@echo "$(BLUE)Updating container apps...$(RESET)"
	@$(MAKE) azure-update-ml
	@$(MAKE) azure-update-dashboard
	@echo ""
	@$(MAKE) azure-urls
	@echo "$(GREEN)Deployment complete!$(RESET)"

azure-sync-env: ## Sync environment variables from .env to both container apps
	@echo "$(BLUE)Syncing environment variables to container apps...$(RESET)"
	@$(AZURE_PREFLIGHT) env -q || exit 1
	@echo ""
	@echo "$(YELLOW)Updating ml-model env vars...$(RESET)"
	@az containerapp update \
		--name ml-model \
		--resource-group $(RESOURCE_GROUP) \
		--set-env-vars \
			PYTHONUNBUFFERED=1 \
			ENVIRONMENT=production \
			MONGO_URI=$(MONGO_URI) \
			MONGO_HOST=$(COSMOSDB_HOST) \
			MONGO_PORT=$(COSMOSDB_PORT) \
			MONGO_DATABASE=$(COSMOSDB_DATABASE) \
			MONGO_USER=$(COSMOSDB_USER) \
			MONGO_PASSWORD=$(COSMOSDB_PASSWORD) \
			MONGO_TLS=true \
			MONGO_AUTH_SOURCE=admin \
		--output none
	@echo "$(GREEN)ml-model env vars updated$(RESET)"
	@echo ""
	@echo "$(YELLOW)Updating dashboard env vars...$(RESET)"
	@ML_URL=$$(az containerapp show --name ml-model --resource-group $(RESOURCE_GROUP) \
		--query "properties.configuration.ingress.fqdn" --output tsv); \
	az containerapp update \
		--name dashboard \
		--resource-group $(RESOURCE_GROUP) \
		--set-env-vars \
			PYTHONUNBUFFERED=1 \
			ENVIRONMENT=production \
			MONGO_URI=$(MONGO_URI) \
			MONGO_HOST=$(COSMOSDB_HOST) \
			MONGO_PORT=$(COSMOSDB_PORT) \
			MONGO_DATABASE=$(COSMOSDB_DATABASE) \
			MONGO_USER=$(COSMOSDB_USER) \
			MONGO_PASSWORD=$(COSMOSDB_PASSWORD) \
			MONGO_TLS=true \
			MONGO_AUTH_SOURCE=admin \
			ML_SERVICE_URL=http://$$ML_URL \
		--output none
	@echo "$(GREEN)dashboard env vars updated$(RESET)"
	@echo ""
	@echo "$(GREEN)Environment variables synced successfully!$(RESET)"

azure-status: ## Show status of Azure container apps
	@echo "$(BLUE)Container Apps Status:$(RESET)"
	@$(AZURE_PREFLIGHT) env -q || exit 1
	@az containerapp list \
		--resource-group $(RESOURCE_GROUP) \
		--output table 2>/dev/null || echo "  No apps found or not logged in"

azure-urls: ## Show dashboard and ml-model URLs
	@echo "$(BLUE)Application URLs:$(RESET)"
	@$(AZURE_PREFLIGHT) env -q || exit 1
	@ML_URL=$$(az containerapp show --name ml-model --resource-group $(RESOURCE_GROUP) \
		--query "properties.configuration.ingress.fqdn" --output tsv 2>/dev/null); \
	if [ -n "$$ML_URL" ]; then \
		echo "  ML-Model (internal): http://$$ML_URL"; \
	else \
		echo "  ML-Model: Not deployed"; \
	fi
	@DASH_URL=$$(az containerapp show --name dashboard --resource-group $(RESOURCE_GROUP) \
		--query "properties.configuration.ingress.fqdn" --output tsv 2>/dev/null); \
	if [ -n "$$DASH_URL" ]; then \
		echo "  Dashboard (public):  https://$$DASH_URL"; \
	else \
		echo "  Dashboard: Not deployed"; \
	fi

azure-env-vars: ## Show environment variables from both container apps
	@echo "$(BLUE)Container Apps Environment Variables:$(RESET)"
	@$(AZURE_PREFLIGHT) env -q || exit 1
	@echo ""
	@echo "$(YELLOW)ML-Model:$(RESET)"
	@az containerapp show \
		--name ml-model \
		--resource-group $(RESOURCE_GROUP) \
		--query "properties.template.containers[0].env" \
		--output table 2>/dev/null || echo "  Not deployed"
	@echo ""
	@echo "$(YELLOW)Dashboard:$(RESET)"
	@az containerapp show \
		--name dashboard \
		--resource-group $(RESOURCE_GROUP) \
		--query "properties.template.containers[0].env" \
		--output table 2>/dev/null || echo "  Not deployed"

azure-logs-ml: ## Stream ml-model container logs
	@$(AZURE_PREFLIGHT) env -q || exit 1
	@echo "$(BLUE)Streaming ml-model logs (Ctrl+C to stop)...$(RESET)"
	@az containerapp logs show \
		--name ml-model \
		--resource-group $(RESOURCE_GROUP) \
		--follow

azure-logs-dashboard: ## Stream dashboard container logs
	@$(AZURE_PREFLIGHT) env -q || exit 1
	@echo "$(BLUE)Streaming dashboard logs (Ctrl+C to stop)...$(RESET)"
	@az containerapp logs show \
		--name dashboard \
		--resource-group $(RESOURCE_GROUP) \
		--follow
