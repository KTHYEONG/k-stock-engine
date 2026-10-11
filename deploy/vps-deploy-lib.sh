# VPS shared deploy library (contract VPS_DEPLOY_CONTRACT_VERSION=1).
# Canonical location: krx-alpha/deploy/host/vps-deploy-lib.sh.
# Vendored copies in other projects must stay byte-identical to this file.
# Sourced bash library, not executable standalone: `source vps-deploy-lib.sh`
# then call vps_deploy_init before any other function. Every stdout log line
# uses the [SYS] category with key=value fields so runner output stays public
# and self-describing without log access.

VPS_DEPLOY_CONTRACT_VERSION=1

declare -gA _VPS_BASE_STARTED=()
declare -gA _VPS_BASE_RESTARTS=()

_VPS_INSPECT_FORMAT='{{.State.Status}}|{{.State.ExitCode}}|{{.State.OOMKilled}}|{{.State.Error}}|{{.State.StartedAt}}|{{.State.FinishedAt}}|{{.RestartCount}}'

_vps_state_root() {
  printf '%s' "${VPS_STATE_ROOT:-$HOME/.local/state/vps-deploy}"
}

_vps_lock_file() {
  printf '%s/image.lock' "$(_vps_state_root)"
}

_vps_evidence_dir() {
  printf '%s/evidence/%s' "$(_vps_state_root)" "${VPS_PROJECT:?VPS_PROJECT is not set}"
}

_vps_evidence_file() {
  printf '%s/%s-%s.log' "$(_vps_evidence_dir)" "${VPS_DEPLOY_START_UTC:?}" "${VPS_SHA12:?}"
}

# Best-effort evidence collection. Writes the full record to the evidence file
# and prints only state summaries to stdout. Always returns 0 so the original
# failure code is never masked. Never calls inspect without --format: full
# inspect output carries Config.Env credentials.
_vps_collect_evidence() {
  local rc="$1" reason="$2"
  local ev names entry out status code oom errmsg started finished restarts
  ev=$(_vps_evidence_file)
  (umask 077 && : >"$ev") 2>/dev/null || true
  {
    printf '[SYS] evidence_collect start project=%s sha=%s stage=%s rc=%s reason=%s deploy_start=%s\n' \
      "${VPS_PROJECT}" "${VPS_SHA12}" "${VPS_STAGE}" "$rc" "$reason" "${VPS_DEPLOY_START_UTC}"
    printf '## docker ps -a\n'
    docker ps -a 2>&1 || true
  } >>"$ev" 2>&1 || true
  names=$(docker ps -a --format '{{.Names}}' 2>/dev/null || true)
  for entry in $names; do
    [ -n "$entry" ] || continue
    out=$(docker inspect --format "$_VPS_INSPECT_FORMAT" "$entry" 2>/dev/null || true)
    IFS='|' read -r status code oom errmsg started finished restarts <<<"$out"
    {
      printf '## inspect %s\n%s\n' "$entry" "$out"
      printf '## logs %s\n' "$entry"
      docker logs --tail 100 "$entry" 2>&1 || true
    } >>"$ev" 2>&1 || true
    printf '[SYS] container=%s status=%s exit=%s oom=%s restarts=%s project=%s sha=%s\n' \
      "$entry" "${status:-unknown}" "${code:-?}" "${oom:-?}" "${restarts:-?}" "${VPS_PROJECT}" "${VPS_SHA12}"
  done
  {
    printf '## journalctl\n'
    journalctl -u docker -u containerd --since "${VPS_DEPLOY_START_UTC}" 2>&1 || true
    printf '## docker system df\n'
    docker system df 2>&1 || true
  } >>"$ev" 2>&1 || true
  chmod 600 "$ev" 2>/dev/null || true
  return 0
}

# Single failure funnel. Prints exactly one ::error annotation line, then
# exits with the original code. Re-entry (ERR followed by EXIT) is a no-op.
_vps_handle_failure() {
  local rc="$1" lineno="$2" reason="$3"
  local ev
  if [ "${_vps_fail_handled:-0}" -eq 1 ]; then
    exit "${_vps_fail_rc:-$rc}"
  fi
  _vps_fail_handled=1
  _vps_fail_rc=$rc
  trap - ERR EXIT
  set +e
  set +u
  VPS_STAGE="${VPS_STAGE:-init}"
  _vps_collect_evidence "$rc" "$reason"
  ev=$(_vps_evidence_file)
  printf '::error title=vps-deploy/%s/%s::rc=%s line=%s reason=%s sha=%s evidence=%s\n' \
    "${VPS_PROJECT}" "${VPS_STAGE}" "$rc" "$lineno" "$reason" "${VPS_SHA12}" "$ev"
  exit "$rc"
}

_vps_on_err() {
  local lineno="$1" rc="$2"
  _vps_handle_failure "$rc" "$lineno" "${VPS_FAIL_REASON:-error}"
}

_vps_on_exit() {
  local rc="$1"
  if [ "$rc" -ne 0 ] && [ "${_vps_fail_handled:-0}" -eq 0 ]; then
    _vps_handle_failure "$rc" "${LINENO:-0}" "${VPS_FAIL_REASON:-error}"
  fi
}

vps_deploy_init() {
  local project="$1" sha="$2"
  set -Eeuo pipefail
  VPS_PROJECT=$project
  VPS_SHA=$sha
  VPS_SHA12=${sha:0:12}
  VPS_DEPLOY_START_UTC=$(date -u +%Y%m%dT%H%M%SZ)
  VPS_STAGE=init
  VPS_FAIL_REASON=error
  VPS_STAGE_START=$(date +%s)
  _vps_fail_handled=0
  _vps_fail_rc=0
  export VPS_PROJECT VPS_SHA VPS_SHA12 VPS_DEPLOY_START_UTC VPS_STAGE VPS_FAIL_REASON VPS_STAGE_START
  mkdir -p "$(_vps_state_root)" "$(_vps_evidence_dir)"
  trap '_vps_on_err ${LINENO} $?' ERR
  trap '_vps_on_exit $?' EXIT
}

vps_stage() {
  local name="$1" now elapsed
  now=$(date +%s)
  elapsed=$((now - ${VPS_STAGE_START:-$now}))
  printf '[SYS] stage=%s status=ok elapsed_s=%s project=%s sha=%s\n' \
    "${VPS_STAGE:-init}" "$elapsed" "${VPS_PROJECT}" "${VPS_SHA12}"
  VPS_STAGE=$name
  VPS_STAGE_START=$now
  VPS_FAIL_REASON=error
  export VPS_STAGE VPS_STAGE_START VPS_FAIL_REASON
  printf '[SYS] stage=%s status=start project=%s sha=%s\n' "$name" "${VPS_PROJECT}" "${VPS_SHA12}"
}

# Holds the shared image lock inside its own subshell and releases it on
# return. Callers never hold the lock across a waiting gate. Retries the pull
# 3 times (sleeps from VPS_PULL_RETRY_DELAYS, default "10 30") and fails
# closed when the image revision label differs from the requested sha.
vps_pull_verified() {
  local image_repo="$1" sha="$2"
  local lock wait_s rcfile rc reason
  lock=$(_vps_lock_file)
  wait_s=${VPS_IMAGE_LOCK_WAIT_S:-900}
  mkdir -p "$(dirname -- "$lock")"
  rcfile=$(mktemp)
  if (
    set +E
    trap - ERR EXIT
    set +e
    set +u
    exec 200>"$lock"
    if ! flock -s -w "$wait_s" 200; then printf 'image_lock_timeout' >"$rcfile"; exit 10; fi
    delays_str=${VPS_PULL_RETRY_DELAYS:-"10 30"}
    read -r -a delays <<<"$delays_str"
    attempt=0
    while true; do
      attempt=$((attempt + 1))
      if docker pull "${image_repo}:sha-${sha}"; then break; fi
      if [ "$attempt" -gt "${#delays[@]}" ]; then printf 'pull_failed' >"$rcfile"; exit 11; fi
      sleep "${delays[$((attempt - 1))]}"
    done
    label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "${image_repo}:sha-${sha}" 2>/dev/null || true)
    if [ "$label" != "$sha" ]; then printf 'revision_mismatch' >"$rcfile"; exit 12; fi
    exit 0
  ); then
    rm -f "$rcfile"
    return 0
  else
    rc=$?
    reason=$(cat "$rcfile" 2>/dev/null || true)
    rm -f "$rcfile"
    case "$reason" in
      image_lock_timeout | revision_mismatch | pull_failed) ;;
      *) reason=error ;;
    esac
    VPS_FAIL_REASON=$reason
    export VPS_FAIL_REASON
    vps_fail "$reason"
  fi
}

# Holds the shared image lock inside its own subshell and releases it on return.
vps_promote_latest() {
  local image_repo="$1" sha="$2"
  local lock wait_s rcfile rc reason
  lock=$(_vps_lock_file)
  wait_s=${VPS_IMAGE_LOCK_WAIT_S:-900}
  mkdir -p "$(dirname -- "$lock")"
  rcfile=$(mktemp)
  if (
    set +E
    trap - ERR EXIT
    set +e
    set +u
    exec 200>"$lock"
    if ! flock -s -w "$wait_s" 200; then printf 'image_lock_timeout' >"$rcfile"; exit 10; fi
    if ! docker tag "${image_repo}:sha-${sha}" "${image_repo}:latest"; then printf 'promote_failed' >"$rcfile"; exit 11; fi
    exit 0
  ); then
    rm -f "$rcfile"
    return 0
  else
    rc=$?
    reason=$(cat "$rcfile" 2>/dev/null || true)
    rm -f "$rcfile"
    case "$reason" in
      image_lock_timeout | promote_failed) ;;
      *) reason=error ;;
    esac
    VPS_FAIL_REASON=$reason
    export VPS_FAIL_REASON
    vps_fail "$reason"
  fi
}

# Records each container baseline then polls until the window ends. Fails when
# a container stops running, its restart count grows, its start time changes,
# or it was OOM-killed. Poll interval defaults to 5 s (VPS_STABLE_POLL_S).
vps_verify_stable() {
  local window_s="$1"
  shift
  local poll="${VPS_STABLE_POLL_S:-5}"
  local entry out status code oom errmsg started finished restarts
  local base_started base_restarts deadline
  for entry in "$@"; do
    if ! out=$(docker inspect --format "$_VPS_INSPECT_FORMAT" "$entry" 2>/dev/null); then
      vps_fail unstable
    fi
    IFS='|' read -r status code oom errmsg started finished restarts <<<"$out"
    if [ "$status" != "running" ] || [ "$oom" = "true" ]; then
      vps_fail unstable
    fi
    _VPS_BASE_STARTED[$entry]=$started
    _VPS_BASE_RESTARTS[$entry]=$restarts
  done
  deadline=$(($(date +%s) + window_s))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep "$poll"
    for entry in "$@"; do
      if ! out=$(docker inspect --format "$_VPS_INSPECT_FORMAT" "$entry" 2>/dev/null); then
        vps_fail unstable
      fi
      IFS='|' read -r status code oom errmsg started finished restarts <<<"$out"
      if [ "$status" != "running" ] || [ "$oom" = "true" ]; then
        vps_fail unstable
      fi
      base_started=${_VPS_BASE_STARTED[$entry]:-}
      base_restarts=${_VPS_BASE_RESTARTS[$entry]:-}
      if [ "$started" != "$base_started" ] || [ "$restarts" != "$base_restarts" ]; then
        vps_fail unstable
      fi
    done
  done
  printf '[SYS] stage=%s status=ok check=stable window_s=%s containers=%s project=%s sha=%s\n' \
    "${VPS_STAGE}" "$window_s" "$*" "${VPS_PROJECT}" "${VPS_SHA12}"
}

# Explicit failure: annotation, evidence, exit 1.
vps_fail() {
  local reason="${1:-error}"
  VPS_FAIL_REASON=$reason
  export VPS_FAIL_REASON
  _vps_handle_failure 1 "${BASH_LINENO[0]:-0}" "$reason"
}
