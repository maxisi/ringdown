#!/usr/bin/env bash
set -euo pipefail

# Prefetch the data needed by the example notebooks:
# - GWTC-2.1 GW150914 PE samples from Zenodo (fit_from_imr_result.ipynb)
# - GWOSC strain around GW150914, stored in the astropy download cache so
#   that gwpy's fetch_open_data (with GWPY_CACHE=1) finds it and skips the
#   ~500 MB/detector download.
# Requires python with astropy and gwosc installed.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# PE samples (the notebook's `wget -nc` skips the download if this exists)
PE_FILE="$ROOT/docs/examples/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
PE_URL="https://zenodo.org/records/6513631/files/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
if [ -s "$PE_FILE" ]; then
    echo "already present: $(basename "$PE_FILE")"
else
    echo "Downloading $(basename "$PE_FILE")"
    curl --fail --location --retry 5 --retry-delay 10 --retry-all-errors \
        --remove-on-error --connect-timeout 30 --max-time 1800 \
        -o "$PE_FILE" "$PE_URL"
fi

# Resolve the strain file URLs with gwosc, exactly as gwpy will at notebook
# runtime; the gwosc.org metadata API can time out from CI runners, so fall
# back to the known archive URLs for the segment requested by
# fit_from_imr_result.ipynb (4 s at 16 kHz around trigger 1126259462.391).
URLS_FILE="$(mktemp)"
if python - > "$URLS_FILE" <<'EOF'
from gwosc.locate import get_urls
start, end = 1126259460, 1126259465
for ifo in ("H1", "L1"):
    for url in get_urls(ifo, start, end, sample_rate=16384, format="hdf5"):
        print(url)
EOF
then
    echo "resolved strain URLs with gwosc"
else
    echo "gwosc API unreachable; falling back to pinned URLs" >&2
    cat > "$URLS_FILE" <<'EOF'
https://gwosc.org/archive/data/O1_16KHZ/1126170624/H-H1_LOSC_16_V1-1126256640-4096.hdf5
https://gwosc.org/archive/data/O1_16KHZ/1126170624/L-L1_LOSC_16_V1-1126256640-4096.hdf5
EOF
fi

# Download each file with curl (retries + resumable, unlike astropy's own
# downloader) and import it into the astropy cache under its URL key.
TMP_DL="$(mktemp -d)"
trap 'rm -rf "$TMP_DL"' EXIT
while read -r url; do
    [ -z "$url" ] && continue
    if python -c "import sys; from astropy.utils.data import is_url_in_cache; sys.exit(0 if is_url_in_cache(sys.argv[1]) else 1)" "$url"; then
        echo "already cached: $url"
        continue
    fi
    f="$TMP_DL/$(basename "$url")"
    echo "Downloading $url"
    # abort if stalled below 10 kB/s for 60 s, then retry resumes (-C -)
    curl --fail --location --retry 10 --retry-delay 15 --retry-all-errors \
        --connect-timeout 30 --speed-limit 10000 --speed-time 60 \
        --continue-at - -o "$f" "$url"
    python -c "import sys; from astropy.utils.data import import_file_to_cache; import_file_to_cache(sys.argv[1], sys.argv[2]); print('cached:', sys.argv[1])" "$url" "$f"
    rm -f "$f"
done < "$URLS_FILE"
