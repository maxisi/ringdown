#/usr/bin/env bash

mkdir -p data
cd data
for url in \
    https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW150914/v2/H-H1_LOSC_4_V2-1126259446-32.hdf5 \
    https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW150914/v2/L-L1_LOSC_4_V2-1126259446-32.hdf5
do
    f=$(basename "$url")
    if [ ! -f "$f" ]; then
        curl -fsSL -o "$f" "$url"
    fi
done

cd ..

ringdown_pipe etc/ringdown_pipe_example.ini
