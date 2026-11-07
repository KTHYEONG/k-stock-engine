#!/usr/bin/env bash
# Classify whether BASE..HEAD touches anything that reaches the running system.
# Usage: runtime_changed.sh BASE HEAD
#   exit 0  a runtime path changed (image, host units, deploy config, workflows, unknown paths)
#   exit 1  only non-runtime paths changed (tests, docs, markdown, AI tool config)
#   exit 2  the range cannot be determined (missing base, force push, fetch failure)
# Callers decide the safe direction for exit 2; the build/deploy gate treats it as runtime.
set -uo pipefail
NON_RUNTIME='^(tests/|docs/|\.agents/|\.claude/|\.codex/|\.serena/|legacy/)|\.md$'
base="${1:-}"
head="${2:-}"
if [ -z "$base" ] || [ -z "$head" ] || [ -z "${base//0/}" ]; then
  echo "range: unknown (no base)"; exit 2
fi
git fetch --quiet origin "$base" "$head" 2>/dev/null || true
if ! git cat-file -e "$base^{commit}" 2>/dev/null || ! git cat-file -e "$head^{commit}" 2>/dev/null; then
  echo "range: unknown (commits unavailable)"; exit 2
fi
if ! git merge-base --is-ancestor "$base" "$head"; then
  echo "range: unknown (non-linear)"; exit 2
fi
if ! files="$(git diff --name-only "$base" "$head")"; then
  echo "range: unknown (diff failed)"; exit 2
fi
runtime_files="$(printf '%s\n' "$files" | grep -Ev "$NON_RUNTIME" | grep -v '^$' || true)"
if [ -n "$runtime_files" ]; then
  echo "runtime change:"; printf '%s\n' "$runtime_files" | head -20 | sed 's/^/  /'; exit 0
fi
echo "non-runtime only: $(printf '%s\n' "$files" | grep -vc '^$') file(s)"; exit 1
