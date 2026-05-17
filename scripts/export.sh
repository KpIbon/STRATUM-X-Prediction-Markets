#!/bin/bash
# Submission export script
# Creates zip + run command for ProphetArena evaluation

echo "Building STRATUM-X submission package..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE="stratum_x_${TIMESTAMP}"

mkdir -p /tmp/${PACKAGE}
cp -r src /tmp/${PACKAGE}/
cp run.py /tmp/${PACKAGE}/
cp requirements.txt /tmp/${PACKAGE}/
cp README_SUBMISSION.md /tmp/${PACKAGE}/ 2>/dev/null || true
cp -r configs /tmp/${PACKAGE}/ 2>/dev/null || true
cp -r benchmarks /tmp/${PACKAGE}/ 2>/dev/null || true

cd /tmp
zip -r ${PACKAGE}.zip ${PACKAGE}/
cp ${PACKAGE}.zip ~/Downloads/

echo ""
echo "============================================"
echo "  SUBMISSION PACKAGE READY"
echo "============================================"
echo ""
echo "  File: ${PACKAGE}.zip"
echo "  Location: ~/Downloads/${PACKAGE}.zip"
echo ""
echo "  Run command:"
echo "    python run.py --export"
echo ""
echo "  For ProphetArena evaluation:"
echo "    python -c \"from src.api.prophet_arena import ProphetArenaAdapter; a = ProphetArenaAdapter(); a.run_round({})\""
echo ""
echo "============================================"