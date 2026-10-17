#!/usr/bin/env bash
# Install KSE systemd user units and enable timers.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/systemd" && pwd)"
DEST_DIR="${HOME}/.config/systemd/user"

mkdir -p "${DEST_DIR}"
mkdir -p "${HOME}/.cache/kis"
chmod 700 "${HOME}/.cache/kis"
# Legacy dbsec dir backs the read-only mount; remove with the legacy fallback.
mkdir -p "${HOME}/.cache/dbsec"
chmod 700 "${HOME}/.cache/dbsec"
touch "${HOME}/.cache/kis/.host-admission"
mkdir -p "${HOME}/k-stock-engine/data" "${HOME}/k-stock-engine/logs"
# Alert credentials live in their own 0600 file so only the audit and alert containers receive them.
touch "${HOME}/quant-secrets/kse-alert.env"
chmod 600 "${HOME}/quant-secrets/kse-alert.env"
cp "${SRC_DIR}"/kse-*.timer "${SRC_DIR}"/kse-*.service "${DEST_DIR}/"

systemctl --user daemon-reload

systemctl --user enable --now \
  kse-collect-eve.timer \
  kse-collect-dbsec.timer \
  kse-backup.timer \
  kse-morning.timer \
  kse-auction-open.timer \
  kse-daily-audit.timer \
  kse-weekly-report.timer

loginctl enable-linger "${USER}"
