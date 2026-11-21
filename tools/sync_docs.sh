#!/usr/bin/env bash
# Copy this package's user guide (docs/) into the 32-ID TXM Software User
# Guide, section xanes2d_analysis. The pages here are the source.
#   tools/sync_docs.sh [path/to/32id-gui-docs]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guide="${1:-$here/../32id-gui-docs}"
dest="$guide/docs/xanes2d_analysis"
[ -d "$guide/docs" ] || { echo "not found: $guide/docs" >&2; exit 1; }
mkdir -p "$dest/images"
rsync -a --delete --include='*.md' --exclude='*' "$here/docs/" "$dest/"
rsync -a --delete "$here/docs/images/" "$dest/images/"
echo "synced docs → $dest  (rebuild and commit 32id-gui-docs)"
