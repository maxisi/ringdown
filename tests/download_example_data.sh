#!/usr/bin/env bash
set -euo pipefail

# Shared GW150914 strain files used by the CLI config tests.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data"
cd "$ROOT/data"

urls=(
    https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW150914/v2/H-H1_LOSC_4_V2-1126259446-32.hdf5
    https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW150914/v2/L-L1_LOSC_4_V2-1126259446-32.hdf5
)

for url in "${urls[@]}"; do
    f=$(basename "$url")
    if [ -s "$f" ]; then
        continue
    fi
    echo "Downloading $f"
    curl --fail --location --retry 5 --retry-delay 10 --retry-all-errors \
        --remove-on-error --connect-timeout 30 --max-time 180 -o "$f" "$url"
done
