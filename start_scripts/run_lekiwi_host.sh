#!/bin/bash
# Launch script for LeKiwi host on Raspberry Pi.
# Uses torch_stub/ to bypass incompatible PyTorch SVE instructions on Cortex-A72.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}/torch_stub:${SCRIPT_DIR}/src:${PYTHONPATH}"

echo "Starting LeKiwi host..."
exec python -m lerobot.robots.lekiwi.lekiwi_host "$@"
