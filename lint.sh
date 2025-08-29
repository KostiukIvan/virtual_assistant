#!/bin/bash

# ==============================================================================
# Best Practices Python Linting + Polishing Script
#
# This script performs the following actions:
# 1. Formats Python code with Black.
# 2. Lints with Ruff (errors + autofixes, including import order).
# 3. Checks type consistency with Mypy.
# 4. Runs Bandit for security checks.
# 5. Scans dependencies with Safety (requires requirements.txt).
# 6. Optionally formats docstrings with Docformatter.
# 7. Formats JSON/YAML/Markdown files with Prettier (if installed).
#
# Usage:
#   ./lint.sh
#   ./lint.sh my_specific_directory/
# ==============================================================================

set -e  # Exit on first failure

TARGET_DIR=${1:-"."}

echo "🚀 Starting Python code quality checks on '$TARGET_DIR'..."

# Step 1: Format code with Black
echo -e "\n🎨 Formatting code with Black..."
black "$TARGET_DIR"

# Step 2: Lint + auto-fix with Ruff
echo -e "\n🔍 Linting and auto-fixing with Ruff..."
ruff check "$TARGET_DIR" --fix
ruff check "$TARGET_DIR"

# Step 3: Type checking with Mypy
echo -e "\n🔬 Checking types with Mypy..."
mypy "$TARGET_DIR"

# Step 4: Security linting with Bandit
echo -e "\n🛡️  Running security checks with Bandit..."
bandit -r "$TARGET_DIR" || true  # don't block CI unless you want strictness

# Step 5: Dependency security with Safety (requires requirements.txt or poetry.lock)
if [ -f "requirements.txt" ]; then
  echo -e "\n📦 Checking dependencies with Safety..."
  safety check -r requirements.txt || true
fi

# # Step 6: Auto-format docstrings with Docformatter (optional)
# if command -v docformatter &> /dev/null; then
#   echo -e "\n📝 Formatting docstrings with Docformatter..."
#   docformatter -i -r "$TARGET_DIR"
# fi

# Step 7: Format non-Python files with Prettier (optional)
if command -v prettier &> /dev/null; then
  echo -e "\n🧹 Formatting JSON/YAML/Markdown with Prettier..."
  prettier --write "**/*.{json,yml,yaml,md}"
fi

echo -e "\n🎉 All checks completed!"
