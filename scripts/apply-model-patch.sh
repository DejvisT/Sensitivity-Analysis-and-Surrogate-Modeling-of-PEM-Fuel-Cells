#!/bin/bash
set -e

PATCH_FILE="patches/model-changes.patch"
TARGET_DIR="external/AlphaPEM/model"

echo "Applying patch to $TARGET_DIR..."

(cd "$TARGET_DIR" && patch -p1 < "../../../$PATCH_FILE")

echo "✅ Patch applied successfully."
