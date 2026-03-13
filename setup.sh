#!/usr/bin/env bash
# setup.sh — One-shot setup for the Medical RAG system
# Run: bash setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║      MedQuery RAG — Environment Setup        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
PY=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python: $PY"
MAJOR=$(echo $PY | cut -d. -f1)
MINOR=$(echo $PY | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || [ "$MINOR" -lt 10 ]; then
  echo "  ✗ Python 3.10+ required. Found $PY"
  exit 1
fi
echo "  ✓ Python version OK"

# ── 2. PyMuPDF conflict fix ───────────────────────────────────────────────────
echo ""
echo "  Checking for PyMuPDF conflict..."
if pip show frontend &>/dev/null; then
  echo "  Removing conflicting 'frontend' package..."
  pip uninstall frontend -y --quiet
fi

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "  Installing dependencies..."
pip install -r requirements.txt --quiet
echo "  ✓ Dependencies installed"

# ── 4. Create required directories ───────────────────────────────────────────
echo ""
echo "  Creating directory structure..."
mkdir -p data/raw data/processed data/chunks embeddings/chroma_db logs
echo "  ✓ Directories created"

# ── 5. API key check ──────────────────────────────────────────────────────────
echo ""
if [ -z "$GROQ_API_KEY" ]; then
  echo "  ⚠  GROQ_API_KEY is not set."
  echo ""
  echo "  Get a free API key at: https://console.groq.com"
  echo "  Then set it:"
  echo "    Linux/Mac:  export GROQ_API_KEY='gsk_...'"
  echo "    Windows:    \$env:GROQ_API_KEY='gsk_...'"
else
  echo "  ✓ GROQ_API_KEY is set"
fi

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────────"
echo "  Setup complete. Next steps:"
echo ""
echo "    1. Set GROQ_API_KEY (if not already done)"
echo "    2. Build knowledge base:"
echo "         python pipeline/embedder.py"
echo "    3. Start the API + UI:"
echo "         uvicorn api.main:app --reload --port 8000"
echo "    4. Open: http://localhost:8000"
echo ""
echo "  Optional:"
echo "    python tests/run_all.py --skip-llm   # run all tests"
echo "    python monitoring/dashboard.py        # view metrics"
echo "  ─────────────────────────────────────────────"
echo ""
