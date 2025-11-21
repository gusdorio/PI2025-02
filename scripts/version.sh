#!/bin/bash
# version.sh - Generate version metadata for container images

# Get git commit hash (short form)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

# Get git branch name, sanitize for use in image tags
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null | sed 's/[^a-zA-Z0-9._-]/-/g' || echo "unknown")

# Check if working directory is clean
GIT_DIRTY=$(git diff --quiet 2>/dev/null || echo "-dirty")

# Timestamp in format: YYYYMMDD-HHMMSS
BUILD_TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")

# Semantic version - you'll update this manually for releases
SEMANTIC_VERSION="0.1.0"

# Construct full version tag
# Format: v0.1.0-main-abc1234-20241110-142305
if [ "$GIT_BRANCH" = "main" ] || [ "$GIT_BRANCH" = "master" ]; then
    VERSION_TAG="v${SEMANTIC_VERSION}-${GIT_COMMIT}-${BUILD_TIMESTAMP}"
else
    VERSION_TAG="v${SEMANTIC_VERSION}-${GIT_BRANCH}-${GIT_COMMIT}-${BUILD_TIMESTAMP}"
fi

# Export for use in make or other scripts
export VERSION_TAG
export GIT_COMMIT
export GIT_BRANCH
export SEMANTIC_VERSION
export BUILD_TIMESTAMP

# Output version info (useful for debugging)
if [ "$1" = "--show" ]; then
    echo "Semantic Version: ${SEMANTIC_VERSION}"
    echo "Git Commit: ${GIT_COMMIT}${GIT_DIRTY}"
    echo "Git Branch: ${GIT_BRANCH}"
    echo "Build Timestamp: ${BUILD_TIMESTAMP}"
    echo "Full Version Tag: ${VERSION_TAG}"
fi

echo "VERSION_TAG=${VERSION_TAG}"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "BUILD_TIMESTAMP=${BUILD_TIMESTAMP}"