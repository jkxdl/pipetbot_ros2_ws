#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_FILE="${SCRIPT_DIR}/../applications/pipettingrobot_operator.desktop"
TARGET="${HOME}/Desktop/pipettingrobot_operator.desktop"

install -m 755 "${DESKTOP_FILE}" "${TARGET}"
chmod +x "${TARGET}"
echo "Desktop shortcut installed at ${TARGET}"

