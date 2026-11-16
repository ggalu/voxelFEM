#!/usr/bin/env bash
#
# clean_artifacts.sh — remove generated output files from the repo.
#
# Deletes every .vtk, .vti and .npy file under the repository root (the
# per-voxel field exports and intermediate arrays written by the example and
# pipeline scripts). The .git directory is never touched.
#
# Usage:
#   ./clean_artifacts.sh            # delete the artifacts
#   ./clean_artifacts.sh -n         # dry run: list what would be deleted
#   ./clean_artifacts.sh --dry-run  # same as -n
#
set -euo pipefail

# Repo root = directory containing this script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
case "${1:-}" in
    -n|--dry-run) DRY_RUN=1 ;;
    "")           ;;
    *) echo "Unknown option: $1" >&2
       echo "Usage: $0 [-n|--dry-run]" >&2
       exit 2 ;;
esac

# Collect matching files (NUL-delimited, so spaces in names are safe),
# excluding anything inside .git.
mapfile -d '' -t FILES < <(
    find "$ROOT" -type d -name .git -prune -o \
        -type f \( -name '*.vtk' -o -name '*.vti' -o -name '*.npy' \) -print0
)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No .vtk/.vti/.npy artifacts found."
    exit 0
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would remove ${#FILES[@]} artifact(s):"
else
    echo "Removing ${#FILES[@]} artifact(s):"
fi

for f in "${FILES[@]}"; do
    echo "  ${f#"$ROOT"/}"
    [[ $DRY_RUN -eq 0 ]] && rm -f "$f"
done

[[ $DRY_RUN -eq 1 ]] && echo "(dry run — nothing deleted)"
