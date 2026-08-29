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

# GWOSC strain into the astropy download cache, resolving the URLs with
# gwosc exactly as gwpy will at notebook runtime
python - <<'EOF'
from astropy.utils.data import download_file, is_url_in_cache
from gwosc.locate import get_urls

# segment requested by fit_from_imr_result.ipynb: 4 s at 16 kHz centered
# on the GW150914 trigger time from the GWTC-2.1 config (1126259462.391)
start, end = 1126259460, 1126259465
for ifo in ("H1", "L1"):
    for url in get_urls(ifo, start, end, sample_rate=16384, format="hdf5"):
        if is_url_in_cache(url):
            print(f"already cached: {url}")
        else:
            print(f"downloading: {url}")
            download_file(url, cache=True, show_progress=False)
EOF
