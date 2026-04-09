#!/usr/bin/env bash

set -e

echo ""
echo "🚀 Installing DhronAI..."
echo ""

# ----------------------------
# Step 1: Check Python
# ----------------------------
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

echo "✅ Python detected: $($PYTHON --version)"

# ----------------------------
# Step 2: Create venv
# ----------------------------
if [ ! -d "dhron_env" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv dhron_env
else
    echo "⚠️ Virtual environment already exists"
fi

# ----------------------------
# Step 3: Activate venv
# ----------------------------
source dhron_env/bin/activate

# ----------------------------
# Step 4: Install package
# ----------------------------
echo "⬇️ Installing DhronAI..."

WHEEL=$(ls dist/*.whl | head -n 1)

if [ -z "$WHEEL" ]; then
    echo "❌ No .whl file found in dist/"
    exit 1
fi

pip install "$WHEEL"

# ----------------------------
# Step 5: Install dependencies
# ----------------------------
echo "📚 Installing dependencies..."
pip install "dhronai[train]"

# ----------------------------
# Done
# ----------------------------
echo ""
echo "✅ Installation Complete!"
echo ""
echo "👉 Activate: source dhron_env/bin/activate"
echo "👉 Run: dhronai --help"
echo ""