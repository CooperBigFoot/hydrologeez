#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: bash init.sh <project-name>"
    echo "Example: bash init.sh my-project"
    exit 1
fi

PROJECT_NAME="$1"
PACKAGE_NAME="${PROJECT_NAME//-/_}"

echo "Initializing project: $PROJECT_NAME (package: $PACKAGE_NAME)"

# Rename placeholder package
mv "src/mypackage" "src/$PACKAGE_NAME"

# Update pyproject.toml
sed -i '' "s/name = \"mypackage\"/name = \"$PROJECT_NAME\"/" pyproject.toml
sed -i '' "s/description = \"Add your description here\"/description = \"$PROJECT_NAME\"/" pyproject.toml
sed -i '' "s|src/mypackage/|src/$PACKAGE_NAME/|" pyproject.toml
sed -i '' "s|src/mypackage/|src/$PACKAGE_NAME/|" CLAUDE.md

# Update CLAUDE.md project overview
sed -i '' "s/SHORT PROJECT DESCRIPTION/$PROJECT_NAME/" CLAUDE.md

# Reset README
cat > README.md << EOF
# $PROJECT_NAME
EOF

# Sync dependencies with new package name
uv sync

# Self-destruct
rm -- "$0"

echo "Done! Project '$PROJECT_NAME' is ready."
