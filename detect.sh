#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./detect.sh <config.yaml>"
    exit 1
fi
VENV_PYTHON="venv/Scripts/python.exe"
if [ ! -f "$VENV_PYTHON" ]; then VENV_PYTHON="python"; fi
$VENV_PYTHON src/detector.py --config "$1"
