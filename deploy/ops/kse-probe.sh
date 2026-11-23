#!/usr/bin/env bash
# Usage: kse-probe.sh MORNING|OPEN|EOD|WEEKLY [YYYY-MM-DD]
# Host pass collects journal/systemd/docker/rclone facts; the probe itself runs read-only in the kse image.
set -euo pipefail

usage() {
    echo "usage: kse-probe.sh MORNING|OPEN|EOD|WEEKLY [YYYY-MM-DD]" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
WIN="$1"
DAY="${2:-}"
case "$WIN" in
    MORNING | OPEN | EOD | WEEKLY) ;;
    *) usage ;;
esac
if [[ -n "$DAY" && ! "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    usage
fi

OPS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${KSE_IMAGE:-ghcr.io/kthyeong/k-stock-engine:latest}"
DATA="${KSE_DATA_DIR:-$HOME/k-stock-engine/data}"
TMP="$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/kse-probe.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

nice -n 19 python3 "$OPS/kse_probe_host.py" "$WIN" "$DAY" >"$TMP/facts.tar"

docker run --rm --network none --memory 1g --cpus 1 \
    -v "$DATA:/data:ro" \
    -v "$OPS:/ops:ro" \
    -v "$TMP:/host:ro" \
    -e KSE_PROBE_FACTS=/host/facts.tar \
    -e TZ=Asia/Seoul \
    --entrypoint /app/.venv/bin/python \
    "$IMAGE" /ops/kse_probe.py "$WIN" "$DAY"
