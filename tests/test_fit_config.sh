#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/download_example_data.sh"

ringdown_fit --verbose --force etc/ringdown_fit_example.ini
