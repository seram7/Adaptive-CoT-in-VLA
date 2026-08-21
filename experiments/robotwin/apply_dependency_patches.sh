#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/RoboTwin /absolute/path/to/ZR-0" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOTWIN_ROOT=$(realpath "$1")
ZR0_ROOT=$(realpath "$2")
ROBOTWIN_REVISION=13c3c47ff4312dd62484bcd51be034af55c062d1
ZR0_REVISION=b1440d4cf27624da2b1aa31268637cf46601c15d

apply_one() {
  local root="$1" expected_revision="$2" patch_file="$3" label="$4"
  local actual_revision
  actual_revision=$(git -C "$root" rev-parse HEAD)
  if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "$label must be checked out at $expected_revision (found $actual_revision)" >&2
    exit 2
  fi
  if git -C "$root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "$label patch already applied"
    return 0
  fi
  if [[ -n "$(git -C "$root" status --porcelain)" ]]; then
    echo "$label worktree must be clean before patching: $root" >&2
    git -C "$root" status --short >&2
    exit 2
  fi
  git -C "$root" apply --check "$patch_file"
  git -C "$root" apply "$patch_file"
  echo "Applied $label patch"
}

apply_one "$ROBOTWIN_ROOT" "$ROBOTWIN_REVISION" \
  "$SCRIPT_DIR/patches/robotwin_13c3c47_lightweight_farmass.patch" RoboTwin
apply_one "$ZR0_ROOT" "$ZR0_REVISION" \
  "$SCRIPT_DIR/patches/zr0_b1440d4_robotwin_server.patch" ZR-0
