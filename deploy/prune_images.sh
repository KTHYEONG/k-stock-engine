#!/usr/bin/env bash
# Untag superseded sha-* images of the KSE repo; best-effort, never removes by image ID or prunes globally.
set -uo pipefail

REPO="${KSE_IMAGE_REPO:-ghcr.io/kthyeong/k-stock-engine}"
KEEP="${KSE_IMAGE_KEEP:-5}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) printf '[SYS] image_prune status=USAGE\n' >&2; exit 64 ;;
  esac
done

if [[ ! "$KEEP" =~ ^[1-9][0-9]*$ ]]; then
  printf '[SYS] image_prune status=USAGE\n' >&2
  exit 64
fi

skip() {
  printf '[SYS] image_prune status=SKIPPED reason=%s\n' "$1"
  exit 0
}

command -v docker >/dev/null 2>&1 || skip docker_unavailable
images=$(docker image ls --no-trunc --format '{{.Repository}} {{.Tag}} {{.ID}}' 2>/dev/null) \
  || skip docker_unavailable

latest_id=""
cand_tags=()
cand_ids=()
while read -r repo tag id; do
  [[ "$repo" == "$REPO" ]] || continue
  if [[ "$tag" == latest ]]; then
    latest_id="$id"
    continue
  fi
  [[ "$tag" =~ ^sha-[0-9a-f]{40}$ ]] || continue
  cand_tags+=("$tag")
  cand_ids+=("$id")
done <<< "$images"

rows=""
for i in "${!cand_tags[@]}"; do
  created=$(docker image inspect --format '{{.Created}}' "$REPO:${cand_tags[$i]}" 2>/dev/null) \
    || skip inspect_failed
  rows+="$created"$'\t'"${cand_tags[$i]}"$'\t'"${cand_ids[$i]}"$'\n'
done

declare -A protected=()
if [[ -n "$latest_id" ]]; then
  protected[$latest_id]=1
fi

container_ids=$(docker ps -a -q --no-trunc 2>/dev/null) || skip container_list_failed
while read -r cid; do
  [[ -n "$cid" ]] || continue
  image_id=$(docker inspect --format '{{.Image}}' "$cid" 2>/dev/null) || skip container_inspect_failed
  protected[$image_id]=1
done <<< "$container_ids"

ranked=$(printf '%s' "$rows" | sort -t $'\t' -k1,1r -k2,2)
rank=0
while IFS=$'\t' read -r _created tag id; do
  [[ -n "$tag" ]] || continue
  rank=$((rank + 1))
  if (( rank <= KEEP )); then
    protected[$id]=1
  fi
done <<< "$ranked"

kept=0
removed=0
failed=0
while IFS=$'\t' read -r _created tag id; do
  [[ -n "$tag" ]] || continue
  if [[ -n "${protected[$id]:-}" ]]; then
    kept=$((kept + 1))
    continue
  fi
  if (( DRY_RUN )); then
    printf '[SYS] image_prune action=would_remove tag=%s\n' "$tag"
    continue
  fi
  if docker rmi "$REPO:$tag" >/dev/null; then
    removed=$((removed + 1))
    printf '[SYS] image_prune action=removed tag=%s\n' "$tag"
  else
    failed=$((failed + 1))
    printf '[SYS] image_prune action=failed tag=%s\n' "$tag"
  fi
done <<< "$ranked"

printf '[SYS] image_prune status=OK kept=%d removed=%d failed=%d\n' "$kept" "$removed" "$failed"
exit 0
