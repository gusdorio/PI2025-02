# Azure CLI Commands Reference - PI2025-02 Project

This document consolidates all Azure CLI commands used for the PI2025-02 cloud data systems project deployment.

## Quick Reference

```bash
az account list --output table
```

## Creation of Service Principal (for GitHub Actions)
GitHub Actions needs to authenticate to two Azure services:

GitHub Actions → Azure Container Registry (to push images) → Azure Container Apps (to deploy)

### Step 1: Create Service Principal for GitHub Actions

```bash
# Create SP with Contributor role scoped to your resource group
az ad sp create-for-rbac \
  --name "github-actions-pi2025" \
  --role contributor \
  --scopes /subscriptions/${AZ_SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}

# Assign AcrPush role to the service principal
az role assignment create \
  --assignee ${AZ_SP_ID} \
  --role AcrPush \
  --scope $ACR_ID
```

### Step 2: Configure GitHub Secrets

In your repository: **Settings → Secrets and variables → Actions**

Add the following secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `AZURE_CREDENTIALS` | Full JSON output from `az ad sp create-for-rbac --sdk-auth` | Service Principal authentication for GitHub Actions |
| `ACR_LOGIN_SERVER` | `<your-acr>.azurecr.io` | Azure Container Registry login server |
| `ACR_USERNAME` | Service Principal App ID | ACR authentication username |
| `ACR_PASSWORD` | Service Principal password | ACR authentication password |
| `COSMOSDB_PASSWORD` | Your CosmosDB primary key | MongoDB connection authentication |

For `AZURE_CREDENTIALS` the final string must be:
{"clientId": "<client_id>", "clientSecret": "<client_secret>", "subscriptionId": "<subscription_id>", "tenantId": "<tenant_id>"}


## Environment Variables Setup

Ensure to set the basic variables at .env before proceeding

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Container Apps Environment                           │
│                              (pi2502-env)                                    │
│                                                                              │
│   ┌────────────────────┐   HTTP (internal)   ┌────────────────────┐          │
│   │     dashboard      │ ─────────────────▶  │     ml-model      │          │
│   │    (external)      │                     │    (internal)      │          │
│   │   port: 8501       │                     │   port: 5000       │          │
│   │ 0.5 CPU / 1.0 Gi   │                     │ 1.0 CPU / 2.0 Gi   │          │
│   └──────────┬─────────┘                     └──────────┬─────────┘          │
│              │                                          │                    │
└──────────────┼──────────────────────────────────────────┼────────────────────┘
               │                                          │
               │              MongoDB API                 │
               └─────────────────┬────────────────────────┘
                                 │
                                 ▼
                   ┌─────────────────────────────┐
                   │      Azure CosmosDB         │
                   │ pi2025.mongo.cosmos.azure   │
                   │       .com:10255            │
                   │   (Serverless / MongoDB)    │
                   └─────────────────────────────┘
                                 ▲
┌────────────────┐              │
│     Users      │──────────────┘
│   (Browser)    │  HTTPS via external ingress
└────────────────┘  https://dashboard.<env>.eastus2.azurecontainerapps.io
```

**Network Connectivity:**

| Source | Destination | Protocol | Access Type |
|--------|-------------|----------|-------------|
| Internet | Dashboard | HTTPS | External ingress (public) |
| Dashboard | ML-Model | HTTP | Internal FQDN (private) |
| Dashboard | CosmosDB | MongoDB (TLS) | Azure backbone |
| ML-Model | CosmosDB | MongoDB (TLS) | Azure backbone |

---

## Phase 1: CosmosDB (MongoDB API)

### 1.1 Create CosmosDB Account

```bash
az cosmosdb create \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --kind MongoDB \
  --server-version "4.2" \
  --default-consistency-level "Session" \
  --locations regionName=$LOCATION failoverPriority=0 isZoneRedundant=false \
  --capabilities EnableServerless
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--kind MongoDB` | MongoDB API | PyMongo compatibility |
| `--server-version "4.2"` | MongoDB 4.2 | Feature/compatibility balance |
| `--default-consistency-level "Session"` | Session | User sees own writes |
| `--capabilities EnableServerless` | Serverless tier | Pay-per-request (cost-effective for academic projects) |

### 1.2 Retrieve Credentials

```bash
# Get primary key
az cosmosdb keys list \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --type keys \
  --query "primaryMasterKey" \
  --output tsv

# Get full connection string
az cosmosdb keys list \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --type connection-strings \
  --query "connectionStrings[0].connectionString" \
  --output tsv
```

### 1.3 Create Database and Collections

```bash
# Create database
az cosmosdb mongodb database create \
  --account-name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --name $COSMOSDB_DATABASE

# Create collections
az cosmosdb mongodb collection create \
  --account-name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --database-name $COSMOSDB_DATABASE \
  --name datasets

az cosmosdb mongodb collection create \
  --account-name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --database-name $COSMOSDB_DATABASE \
  --name pipeline_runs

az cosmosdb mongodb collection create \
  --account-name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --database-name $COSMOSDB_DATABASE \
  --name ml_results
```

### 1.4 Configure Network Access (Firewall)

```bash
# Enable public network access
az cosmosdb update \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --enable-public-network true \
  --public-network-access Enabled

# Allow Azure services (0.0.0.0 = Azure internal services)
az cosmosdb update \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --ip-range-filter "0.0.0.0"
```

**Alternative (tighter security):** Add specific Container Apps outbound IPs:

```bash
# Get Container Apps Environment static IP
STATIC_IP=$(az containerapp env show \
  --name $ENVIRONMENT_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.staticIp" \
  --output tsv)

# Add to CosmosDB firewall
az cosmosdb update \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --ip-range-filter "$STATIC_IP"
```

### 1.5 Verify CosmosDB Status

```bash
# Check account status
az cosmosdb show \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query "{name:name, status:provisioningState, endpoint:documentEndpoint}"

# Check firewall configuration
az cosmosdb show \
  --name $COSMOSDB_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query "{publicNetworkAccess:publicNetworkAccess, ipRules:ipRules, virtualNetworkRules:virtualNetworkRules}" \
  --output json
```

---

## Phase 2: Container Apps Environment

### 2.1 Create Environment

```bash
az containerapp env create \
  --name $ENVIRONMENT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

The environment provides:
- Internal DNS for service-to-service communication
- Shared Log Analytics workspace
- Network isolation boundary

### 2.2 Verify Environment

```bash
az containerapp env show \
  --name $ENVIRONMENT_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "{name:name, status:provisioningState, staticIp:properties.staticIp}" \
  --output json
```

---

## Phase 3: Container Deployment

### 3.1 Deploy ML-Model Service (Internal)

```bash
az containerapp create \
  --name ml-model \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image "${ACR_LOGIN_SERVER}/pi2025-02/ml-model:latest" \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 5000 \
  --ingress internal \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 1 \
  --max-replicas 3 \
  --env-vars \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    MONGO_HOST=$COSMOSDB_HOST \
    MONGO_PORT=$COSMOSDB_PORT \
    MONGO_DATABASE=$COSMOSDB_DATABASE \
    MONGO_USER=$COSMOSDB_USER \
    MONGO_PASSWORD=$COSMOSDB_PASSWORD \
    MONGO_TLS=true \
    MONGO_AUTH_SOURCE=admin
```

| Setting | Value | Reason |
|---------|-------|--------|
| `--ingress internal` | Internal only | Not accessible from internet |
| `--target-port 5000` | ML server port | Matches server.py configuration |
| `--min-replicas 1` | Always running | Avoids cold start for ML processing |
| `--cpu 1.0 --memory 2.0Gi` | Higher resources | ML processing requirements |

### 3.2 Get ML-Model Internal URL

```bash
# Get internal FQDN (required for dashboard configuration)
az containerapp show \
  --name ml-model \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv
```

**URL Format:** `ml-model.internal.<env-id>.eastus2.azurecontainerapps.io`

The `.internal.` segment is critical for internal routing.

### 3.3 Deploy Dashboard Service (External)

```bash
# Get ML model internal URL first
ML_MODEL_URL=$(az containerapp show \
  --name ml-model \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv)

# Deploy dashboard with external ingress
az containerapp create \
  --name dashboard \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image "${ACR_LOGIN_SERVER}/pi2025-02/dashboard:latest" \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8501 \
  --ingress external \
  --cpu 0.5 \
  --memory 1.0Gi \
  --min-replicas 1 \
  --max-replicas 5 \
  --env-vars \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    MONGO_HOST=$COSMOSDB_HOST \
    MONGO_PORT=$COSMOSDB_PORT \
    MONGO_DATABASE=$COSMOSDB_DATABASE \
    MONGO_USER=$COSMOSDB_USER \
    MONGO_PASSWORD=$COSMOSDB_PASSWORD \
    MONGO_TLS=true \
    MONGO_AUTH_SOURCE=admin \
    ML_SERVICE_URL=http://${ML_MODEL_URL}
```

| Setting | Value | Reason |
|---------|-------|--------|
| `--ingress external` | Public access | Browser access for users |
| `--target-port 8501` | Streamlit port | Default Streamlit server port |
| `ML_SERVICE_URL` | Internal FQDN | Dashboard → ML service communication |

### 3.4 Get Dashboard Public URL

```bash
az containerapp show \
  --name dashboard \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv
```

Access via: `https://dashboard.<env-id>.eastus2.azurecontainerapps.io`

---

## Phase 4: Update & Maintenance

### 4.1 Rebuild and Push Images

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Rebuild images (from project root with multi-stage Dockerfile)
docker build --target ml-model -t ${ACR_LOGIN_SERVER}/pi2025-02/ml-model:latest .
docker build --target dashboard -t ${ACR_LOGIN_SERVER}/pi2025-02/dashboard:latest .

# Push to registry
docker push ${ACR_LOGIN_SERVER}/pi2025-02/ml-model:latest
docker push ${ACR_LOGIN_SERVER}/pi2025-02/dashboard:latest
```

### 4.2 Update Container Apps with New Images

```bash
# Update ML-Model
az containerapp update \
  --name ml-model \
  --resource-group $RESOURCE_GROUP \
  --image "${ACR_LOGIN_SERVER}/pi2025-02/ml-model:latest"

# Update Dashboard
az containerapp update \
  --name dashboard \
  --resource-group $RESOURCE_GROUP \
  --image "${ACR_LOGIN_SERVER}/pi2025-02/dashboard:latest"
```

### 4.3 Update Environment Variables

```bash
# Update single variable
az containerapp update \
  --name dashboard \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars ML_SERVICE_URL=http://${ML_MODEL_URL}

# Update multiple variables
az containerapp update \
  --name dashboard \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars \
    MONGO_HOST=$COSMOSDB_HOST \
    MONGO_PASSWORD=$COSMOSDB_PASSWORD
```

### 4.4 Force Container Restarts

```bash
az containerapp revision restart \
  --name dashboard \
  --resource-group PI2025-02 \
  --revision $(az containerapp revision list --name dashboard --resource-group PI2025-02 --query "[0].name" -o tsv)

az containerapp revision restart \
  --name ml-model \
  --resource-group PI2025-02 \
  --revision $(az containerapp revision list --name ml-model --resource-group PI2025-02 --query "[0].name" -o tsv)
```

---

## Debugging & Monitoring

### View Container Logs

```bash
# Stream logs in real-time
az containerapp logs show \
  --name dashboard \
  --resource-group $RESOURCE_GROUP \
  --follow

az containerapp logs show \
  --name ml-model \
  --resource-group $RESOURCE_GROUP \
  --follow
```

### Check Container Status

```bash
# Get running status
az containerapp show \
  --name ml-model \
  --resource-group $RESOURCE_GROUP \
  --query "properties.runningStatus" \
  --output tsv
```

### Inspect Environment Variables

```bash
az containerapp show \
  --name dashboard \
  --resource-group $RESOURCE_GROUP \
  --query "properties.template.containers[0].env" \
  --output table
```

### List All Container Apps

```bash
az containerapp list \
  --resource-group $RESOURCE_GROUP \
  --output table
```

---

## Quick Reference

### Common Troubleshooting Commands

```bash
# Check if services are running
az containerapp show --name ml-model --resource-group $RESOURCE_GROUP --query "properties.runningStatus"
az containerapp show --name dashboard --resource-group $RESOURCE_GROUP --query "properties.runningStatus"

# Verify internal URLs (watch for .internal. segment)
az containerapp show --name ml-model --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn"

# Check CosmosDB firewall status
az cosmosdb show --name $COSMOSDB_ACCOUNT --resource-group $RESOURCE_GROUP --query "publicNetworkAccess"

# Get Container Apps environment outbound IP
az containerapp env show --name $ENVIRONMENT_NAME --resource-group $RESOURCE_GROUP --query "properties.staticIp"
```

### URL Format Reference

| Service | Internal URL Pattern | External URL Pattern |
|---------|----------------------|----------------------|
| ML-Model | `ml-model.internal.<env>.eastus2.azurecontainerapps.io` | N/A (internal only) |
| Dashboard | N/A | `dashboard.<env>.eastus2.azurecontainerapps.io` |
| CosmosDB | `pi2025.mongo.cosmos.azure.com:10255` | Same (Azure backbone) |

---

## Connection Requirements Summary

| Connection | Port | Protocol | TLS | Notes |
|------------|------|----------|-----|-------|
| User → Dashboard | 443 | HTTPS | Yes | Container Apps handles TLS termination |
| Dashboard → ML-Model | 5000 | HTTP | No | Internal traffic within environment |
| Dashboard → CosmosDB | 10255 | MongoDB | Yes | `MONGO_TLS=true` required |
| ML-Model → CosmosDB | 10255 | MongoDB | Yes | `MONGO_TLS=true` required |

---