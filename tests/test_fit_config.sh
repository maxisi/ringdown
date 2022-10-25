#/usr/bin/env bash

mkdir -p data
cd data
wget https://www.gw-openscience.org/eventapi/html/O1_O2-Preliminary/GW150914/v2/H-H1_LOSC_4_V2-1126259446-32.hdf5
wget https://www.gw-openscience.org/eventapi/html/O1_O2-Preliminary/GW150914/v2/L-L1_LOSC_4_V2-1126259446-32.hdf5

cd ..

ringdown_fit etc/ringdown_fit_config_example.ini
