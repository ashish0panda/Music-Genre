#!/bin/bash
set -e

echo "🎵 Setting up Music Genre Classifier Frontend..."
echo ""

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required. Install from https://nodejs.org (v18+)"
    exit 1
fi

NODE_VER=$(node -v)
echo "✓ Node.js $NODE_VER found"

echo "→ Installing dependencies..."
npm install

echo ""
echo "✅ Setup complete!"
echo ""
echo "→ Starting dev server on http://localhost:5173"
echo ""
npm run dev
