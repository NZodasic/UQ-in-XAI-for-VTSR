#!/bin/bash
echo "Starting TSR Experiment on Linux..."
PYTHONUNBUFFERED=1 "${PYTHON:-python3}" -u main.py --config configs/config.yaml "$@"
