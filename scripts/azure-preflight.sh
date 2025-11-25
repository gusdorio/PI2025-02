#!/bin/bash
# ============================================================================
# azure-preflight.sh - Pre-flight checks for Azure operations
# ============================================================================
# Validates prerequisites before running Azure deployment commands.
#
# Usage:
#   ./scripts/azure-preflight.sh [check-type] [-q|--quiet]
#
# Check types:
#   env     - Validate environment variables only
#   login   - Check Azure CLI login status
#   acr     - Verify ACR access
#   apps    - Check Container Apps status
#   all     - Run all checks
#
# Options:
#   -q, --quiet   Suppress output (only show errors)
#
# Exit codes:
#   0 - All checks passed
#   1 - Validation failed
# ============================================================================

set -e

# Check for quiet mode
QUIET=false
for arg in "$@"; do
    if [[ "$arg" == "-q" ]] || [[ "$arg" == "--quiet" ]]; then
        QUIET=true
    fi
done

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Print functions (respect quiet mode)
print_success() { [[ "$QUIET" == "true" ]] || echo -e "${GREEN}[OK]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }  # Always show errors
print_warning() { [[ "$QUIET" == "true" ]] || echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_info() { [[ "$QUIET" == "true" ]] || echo -e "${BLUE}[INFO]${NC} $1"; }
print_header() { [[ "$QUIET" == "true" ]] || echo -e "\n${BLUE}=== $1 ===${NC}\n"; }

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Load .env file
load_env() {
    if [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck source=/dev/null
        source "$ENV_FILE"
        set +a
        return 0
    else
        print_error ".env file not found at $ENV_FILE"
        echo "  Copy .env.example to .env and configure your values"
        return 1
    fi
}

# Check single environment variable
check_var() {
    local var_name=$1
    local var_value="${!var_name}"

    if [ -z "$var_value" ]; then
        print_error "Missing: $var_name"
        return 1
    else
        # Mask sensitive values
        if [[ "$var_name" == *"PASSWORD"* ]] || [[ "$var_name" == *"KEY"* ]] || [[ "$var_name" == *"SECRET"* ]]; then
            print_success "$var_name = *****(masked)"
        else
            print_success "$var_name = $var_value"
        fi
        return 0
    fi
}

# Check environment variables
check_env() {
    print_header "Environment Variables Check"

    load_env || return 1

    local errors=0

    print_info "Core Azure:"
    for var in RESOURCE_GROUP LOCATION ENVIRONMENT_NAME; do
        check_var "$var" || ((errors++))
    done

    echo ""
    print_info "Container Registry:"
    for var in ACR_NAME ACR_LOGIN_SERVER ACR_USERNAME ACR_PASSWORD; do
        check_var "$var" || ((errors++))
    done

    echo ""
    print_info "CosmosDB:"
    for var in COSMOSDB_HOST COSMOSDB_PORT COSMOSDB_DATABASE COSMOSDB_USER COSMOSDB_PASSWORD; do
        check_var "$var" || ((errors++))
    done

    echo ""
    if [ $errors -eq 0 ]; then
        print_success "All environment variables validated"
        return 0
    else
        print_error "$errors variable(s) missing"
        return 1
    fi
}

# Check Azure CLI login
check_login() {
    print_header "Azure CLI Login Check"

    if ! command -v az &> /dev/null; then
        print_error "Azure CLI (az) is not installed"
        echo "  Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        return 1
    fi
    print_success "Azure CLI installed"

    local account
    if ! account=$(az account show 2>/dev/null); then
        print_error "Not logged in to Azure CLI"
        echo "  Run: az login"
        return 1
    fi

    local sub_name sub_id
    sub_name=$(echo "$account" | grep -o '"name": "[^"]*"' | head -1 | cut -d'"' -f4)
    sub_id=$(echo "$account" | grep -o '"id": "[^"]*"' | head -1 | cut -d'"' -f4)

    print_success "Logged in to Azure"
    print_info "  Subscription: $sub_name"
    print_info "  ID: $sub_id"

    return 0
}

# Check ACR access
check_acr() {
    print_header "Azure Container Registry Check"

    load_env || return 1

    if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
        print_error "ACR '$ACR_NAME' not found in resource group '$RESOURCE_GROUP'"
        return 1
    fi

    print_success "ACR exists: $ACR_NAME"
    print_info "  Login server: $ACR_LOGIN_SERVER"

    return 0
}

# Check Container Apps
check_apps() {
    print_header "Container Apps Check"

    load_env || return 1

    local warnings=0

    for app in "ml-model" "dashboard"; do
        if az containerapp show --name "$app" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
            local status
            status=$(az containerapp show --name "$app" --resource-group "$RESOURCE_GROUP" \
                --query "properties.runningStatus" --output tsv 2>/dev/null)
            print_success "App '$app' exists (status: $status)"
        else
            print_warning "App '$app' not deployed"
            ((warnings++))
        fi
    done

    if [ $warnings -gt 0 ]; then
        echo ""
        print_info "Some apps not deployed. Run initial deployment from azure-cli-commands-reference.md"
    fi

    return 0
}

# Run all checks
check_all() {
    local errors=0

    check_env || ((errors++))
    check_login || ((errors++))
    check_acr || ((errors++))
    check_apps || true  # Don't fail on missing apps

    echo ""
    if [ $errors -eq 0 ]; then
        print_success "All pre-flight checks passed!"
        return 0
    else
        print_error "$errors check(s) failed"
        return 1
    fi
}

# Main
case "${1:-all}" in
    env)
        check_env
        ;;
    login)
        check_login
        ;;
    acr)
        check_login && check_acr
        ;;
    apps)
        check_login && check_apps
        ;;
    all)
        check_all
        ;;
    *)
        echo "Usage: $0 {env|login|acr|apps|all}"
        exit 1
        ;;
esac
