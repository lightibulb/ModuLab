#!/usr/bin/env bash
# One-command reproduction: regenerate RESULTS.md and figure
set -euo pipefail
source .venv/bin/activate
python scripts/gen_synthetic.py
pytest -q tests/ --timeout=180
modulab bench
echo "Done. See RESULTS.md"
