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
install -d -m 700 "${HOME}/.local/lib/kse"
_TMP_RUN="$(mktemp "${HOME}/.local/lib/kse/.kse-run.XXXXXX")"
install -m 0755 "${SRC_DIR}/../kse-run" "${_TMP_RUN}"
mv -f "${_TMP_RUN}" "${HOME}/.local/lib/kse/kse-run"
cp "${SRC_DIR}"/kse-*.timer "${SRC_DIR}"/kse-*.service "${DEST_DIR}/"
# The 2026-10-13 OOM hotfix drop-in pinned a local image and bypassed kse-run; its memory fix (7549e46) ships in :latest.
rm -f "${DEST_DIR}/kse-backup.service.d/90-lake-backup-memory.conf"
rmdir "${DEST_DIR}/kse-backup.service.d" 2>/dev/null || true

systemctl --user daemon-reload

mapfile -t timers < <(basename -a "${SRC_DIR}"/kse-*.timer)
systemctl --user enable --now "${timers[@]}"

loginctl enable-linger "${USER}"
