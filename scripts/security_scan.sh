#!/usr/bin/env bash
set -euo pipefail
# Simple SCA using safety (pip install safety)
if ! command -v safety &> /dev/null; then
  pip install safety
fi
safety check --full-report
echo "Security scan completed"
