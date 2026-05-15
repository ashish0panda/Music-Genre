#!/bin/bash
set -e

echo "🎵 Setting up Music Genre Classifier Backend..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install Python 3.9+"
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PYTHON_VER found"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi



echo "→ Installing dependencies (this may take a few minutes first time)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "→ Starting server on http://localhost:8000"
echo "   (First run will download ~500MB model — please wait)"
echo ""
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
