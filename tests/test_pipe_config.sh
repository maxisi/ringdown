#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/download_example_data.sh"

ringdown_pipe etc/ringdown_pipe_example.ini
