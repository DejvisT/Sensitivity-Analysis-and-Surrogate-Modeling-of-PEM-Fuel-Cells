#!/bin/bash
set -e

PATCH_FILE="patches/model-changes.patch"
TARGET_DIR="external/AlphaPEM/model"

echo "Reversing patch from $TARGET_DIR..."

(cd "$TARGET_DIR" && patch -p1 -R < "../../../$PATCH_FILE")

echo "✅ Patch reversed successfully."
