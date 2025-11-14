#!/usr/bin/env bash

# ========= CONFIG (edit these) =========
LIST_FILE="./abhi_list.txt"          # one absolute path per line (dirs or files ok)
REMOTE_USER="cjh9fw"
REMOTE_HOST="portal.cs.virginia.edu"
REMOTE_PORT=22                   # change if non-default
REMOTE_BASE="/p/egoems"  # where to mirror the paths
DRY_RUN=false                    # set to true to preview without copying
# ======================================

# checks
command -v rsync >/dev/null 2>&1 || { echo "rsync not found"; exit 1; }
command -v ssh   >/dev/null 2>&1 || { echo "ssh not found"; exit 1; }
[[ -f "$LIST_FILE" ]] || { echo "List file not found: $LIST_FILE"; exit 1; }

SSH_CMD="ssh -p $REMOTE_PORT"
RSYNC_OPTS=(-aR --partial --info=progress2 --human-readable --protect-args -e "$SSH_CMD")

# dry-run support
if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_OPTS+=(--dry-run)
  echo ">>> DRY RUN enabled (no data will be copied)"
fi

# copy each path, preserving full structure via -R (relative)
# - ignores blank lines and lines starting with '#'
while IFS= read -r SRC || [[ -n "$SRC" ]]; do
  [[ -z "${SRC// }" ]] && continue
  [[ "$SRC" =~ ^# ]] && continue

  if [[ ! -e "$SRC" ]]; then
    echo "WARN: source does not exist locally: $SRC" >&2
    continue
  fi

  echo ">>> Copying: $SRC"
  rsync "${RSYNC_OPTS[@]}" "$SRC" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"
done < "$LIST_FILE"

echo "Done."
