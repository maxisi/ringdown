#!/usr/bin/env python
# coding: utf-8

# Copyright 2022
# Maximiliano Isi <max.isi@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.

import sys
import os
import ringdown as rd
import numpy as np
import argparse
import configparser
import logging
import time
from ast import literal_eval


##############################################################################
# PARSE INPUT
##############################################################################

parser = argparse.ArgumentParser(description="Analyze ringdown of an event "
                                 "at different times.")
parser.add_argument('config', help="Path to configuration file.")
parser.add_argument('-o', '--outdir', default=None, help="Output directory.")
parser.add_argument('--force', action='store_true', help="Run even if output "
                    "directory already exists (pre-existing runs will not be "
                    "repeated).")
parser.add_argument('-s', '--submit', action='store_true', help="Submit slurm "
                    "job automatically.")
parser.add_argument('--nodes', default=20, type=int,
                    help="Maximum number of nodes to request through SLURM.")
parser.add_argument('--seed', default=None, type=int, help="Random seed.")
parser.add_argument('-v', '--verbose', action='store_true')

args = parser.parse_args()

# set up logging and procure Git repo hash
if args.verbose:
    logging.getLogger().setLevel(logging.INFO)

# load config file
config_path = os.path.abspath(args.config)
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"unable to load: {config_path}")
logging.info(f"Loading config from {config_path}")
config = configparser.ConfigParser()
config.read(config_path)

# set random seed (purposedly fail if not provided)
seed = args.seed or config.getint('pipe', 'seed')
logging.info("Random seed set to {}".format(seed))
rng = np.random.default_rng(seed)

# determine run directory
outdir = args.outdir or config.get('pipe', 'outdir', fallback=None)
if not outdir:
    outdir = 'ringdown_pipe_{:.0f}'.format(time.time())
    logging.warning("No run dir provided, defaulting to {}".format(outdir))
outdir = os.path.abspath(outdir)
logging.info("Running in {}".format(outdir))

rerun = False
if os.path.exists(outdir):
    if args.force:
        logging.warning("Run directory already exists.")
        rerun = True
    else:
        raise FileExistsError("Run directory already exists. Exiting.")
else:
    os.makedirs(outdir)

PATHS = {
    'config': 'config.ini',
    'command': 'pipe.sh',
    'run_config': 'engine/{tref:.6f}/{dt0:.6e}/config.ini',
    'run_result': 'engine/{tref:.6f}/{dt0:.6e}/result.nc',
    'run_task': 'TaskFile',
    'exe': 'submit.sh',
    't0': 'start_times.txt',
}
PATHS = {k: os.path.join(outdir, v) for k, v in PATHS.items()}

# record config and arguments for reproducibility
with open(PATHS['config'], 'w') as f:
    config.write(f)

with open(PATHS['command'], 'w') as f:
    f.write("{}\n\n".format(' '.join(sys.argv)))
    for k, v in vars(args).items():
        f.write("# {}: {}\n".format(k, v))
    f.write("\n# {}".format(os.path.realpath(__file__)))
    f.write("\n# ringdown v {}".format(getattr(rd, '__version__', 'unknown')))

# Identify target analysis times. There will be three possibilities:
#     1- listing the times explicitly
#     2- listing time differences with respect to a reference time
#     3- providing start, stop, step instructions to construct start times
#        (potentially relative to a reference time)
# Time steps/differences can be specified in seconds or M, if a reference mass
# is provided (in solar masses).

# Define valid options to specify the start times
T0_SECT = "pipe"
T0_KEYS = {
    'ref': 't0-ref',
    'delta': 't0-delta-list',
    'step': 't0-step',
    'start': 't0-start',
    'stop': 't0-stop'
}
start_stop_step = [T0_KEYS[k] for k in ['start', 'stop', 'step']]

# First make sure that only compatible t0 options were provided
incompatible_sets = [[k, 'delta'] for k in ['start', 'stop', 'step']]
for bad_set in incompatible_sets:
    opt_names = [T0_KEYS[k] for k in bad_set]
    if all([k in config[T0_SECT] for k in opt_names]):
        raise ValueError("incompatible T0 options: {}".format(opt_names))

# Look for a reference mass, to be used when stepping in time
m_ref = config.getfloat(T0_SECT, 'M-ref', fallback=None)
if m_ref:
    # reference time translating from solar masses
    tm_ref = m_ref * rd.qnms.T_MSUN
    logging.info("Reference mass: {} Msun ({} s)".format(m_ref, tm_ref))
else:
    # no reference mass provided, so will default to seconds
    tm_ref = 1

if config.has_option("pipe", "nref"):
    # Select analysis times randomly
    if config.has_option("pipe", T0_KEYS['ref']):
        raise ValueError(f"incompatible options: 'nref', {T0_KEYS['ref']}")

    nref = config.getint('pipe', 'nref')
    logging.info(f"Selecting {nref} reference times.")

    # Determine boundaries of injection region: an interval determined by the
    # greatest and smallest time stamps in data, with an exclusion zone around
    # the indicated trigger time (potentially of zero width, if no exclusion
    # required, i.e., `safe_zone_duration = 0`)
    safe_zone_duration = config.getfloat(
        'pipe', 'safe-zone-duration', fallback=0)
    inj_zone_duration = config.getfloat(
        'pipe', 'inj-zone-duration', fallback=np.inf)

    # Create a fit object, which we will use to manipulate data and obtain
    # prior this fit will automatically take care of all data handling based
    # on config
    fit = rd.Fit.from_config(config, no_cond=True)

    # Pick as many random times uniformly within the injection region as the
    # requested number of injections. Do this in two steps: first, select
    # randomly from before and after the censored trigger time; then, choose
    # randomly between before and after for each entry.
    #
    # NOTE: this selects different times for each detector, but only the ones
    # for the first detector will be used unless the `timeslides` option is
    # given.
    t = fit.data[fit.ifos[0]].time
    t_min = max(fit.t0 - inj_zone_duration, min(t))
    t_max = min(fit.t0 + inj_zone_duration, max(t))

    t_before = rng.uniform(t_min, fit.t0 - safe_zone_duration, nref)
    t_after = rng.uniform(fit.t0 + safe_zone_duration, t_max, nref)
    mask = rng.integers(0, 2, nref)
    t0refs = mask*t_before + (1 - mask)*t_after
else:
    # Look for reference time to be used to construct start times
    t0refs = literal_eval(config.get(T0_SECT, T0_KEYS['ref'], fallback=0))
    try:
        # see if a single reference time was provided
        t0refs = np.array([int(t0refs)])
    except TypeError:
        # multiple reference times provided
        t0refs = np.array(t0refs)

# Now we can safely interpret the options assuming one of three cases
t0_dict = {}
for t0ref in t0refs:
    if T0_KEYS['delta'] in config[T0_SECT]:
        dt0s = np.array(literal_eval(config.get(T0_SECT, T0_KEYS['delta'])))
        t0s = dt0s*tm_ref + t0ref
    elif any([k in config[T0_SECT] for k in start_stop_step]):
        if not all([k in config[T0_SECT] for k in start_stop_step]):
            missing = [k for k in start_stop_step if k not in config[T0_SECT]]
            raise ValueError(
                "missing start/stop/step options: {}".format(missing))
        # add a safety check here, in case the user mistakenly requests
        # stepping based on a GPS time and provides a reference GPS time
        start, stop, step = [config.getfloat(
            T0_SECT, k) for k in start_stop_step]
        if start > 500 and t0ref > 1E8:
            logging.warning("high reference time and stepping start---did you "
                            "accidentally provide GPS times twice?")
        t0s = np.arange(start, stop, step)*tm_ref + t0ref
    else:
        raise ValueError("no timing settings in [{}] section; valid options"
                         " are: {}".format(T0_SECT, list(T0_KEYS.values())))
    t0_dict[t0ref] = t0s


anchor_inj = config.getboolean('pipe', 'anchor-injection', fallback=True)
if anchor_inj and config.has_section('injection'):
    logging.info("Anchoring injections to reference time.")
    if config.has_option('injection', 't0'):
        logging.warning("overwriting injection time.")
    if any(["time" in k for k in config["injection"].keys()]):
        raise ValueError("unsopported injection timing option; use 't0'")

##############################################################################
# SET UP RUNS
##############################################################################

if rerun and os.path.exists(PATHS['run_task']):
    os.remove(PATHS['run_task'])

TASK = "mkdir -p {rundir}; cd {rundir}; ringdown_fit -o {result} {config} "\
       "&> run.log\n"

nruns = sum([len(t0s) for t0s in t0_dict.values()])

logging.info(f"Processing {nruns} runs.")
for tref, t0s in t0_dict.items():
    for i, t0 in enumerate(t0s):
        dt0 = t0 - tref
        # Set up child configuration file
        cpath = PATHS['run_config'].format(i=i, tref=tref, dt0=dt0)
        rundir = os.path.dirname(cpath)
        os.makedirs(rundir, exist_ok=True)
        config_child = configparser.ConfigParser()
        config_child.read(PATHS['config'])

        # Set start time
        config_child['target']['t0'] = str(t0)

        # Set injection time, if anchoring
        if anchor_inj:
            config_child['injection']['t0'] = str(tref)

        # Write out config file for this run
        rpath = PATHS['run_result'].format(i=i, tref=tref, dt0=dt0)
        if os.path.exists(rpath):
            logging.info(f"Run {i} already exists. Will skip.")
        else:
            with open(cpath, 'w') as f:
                config_child.write(f)
            with open(PATHS['run_task'], 'a') as f:
                f.write(TASK.format(rundir=rundir, result=rpath, config=cpath))

logging.info("Done processing {} runs.".format(nruns))


##############################################################################
# SLURM WORKFLOW
##############################################################################

# rusty Broadwell nodes have 28 cores, so each can fit 7 4-core jobs at a time
# given that, make the number of jobs executed at a time a multiple of 7
NSLURM = 7*args.nodes
NCPU = 4

# a user may allocate up to 1280 cores at one time in a center partition, so
# issue a warning if this limit will be exceeded
if NSLURM*NCPU > 1280:
    w = "Requested number of cores ({}) above 1280 user limit."
    logging.warning(w.format(NSLURM*NCPU))

EXE = [
    '#! /usr/bin/env bash',
    'cd {}'.format(outdir),
    'sbatch -p cca -n {} -c {} disBatch {}'.format(
        NSLURM, NCPU, PATHS['run_task']),
    'cd -',
]

epath = PATHS['exe']
with open(epath, 'w') as f:
    f.write('\n'.join(EXE))
st = os.stat(epath)
os.chmod(epath, st.st_mode | 0o111)
if args.submit:
    print("Submitting: {}".format(epath))
    import subprocess
    subprocess.run(epath)
else:
    print("Submit by running: {}".format(epath))
