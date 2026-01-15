#!/bin/bash
# Run all tests for the spotify-tournament project

set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Running fast tests ==="
python -m pytest tests/ -v --ignore=tests/test_model.py -x

echo ""
echo "=== Running slow model tests ==="
python -m pytest tests/test_model.py -v

echo ""
echo "=== All tests passed! ==="
