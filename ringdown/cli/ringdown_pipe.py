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
import numpy as np
import argparse
import configparser
import logging
import time
from ringdown.target import TargetCollection
from ringdown import __version__
import ringdown as rd


##############################################################################
# PARSE INPUT
##############################################################################

def get_parser():
    p = argparse.ArgumentParser(description="Analyze event ringdown at "
                                    "different times.")
    p.add_argument('config', help="Path to configuration file.")
    p.add_argument('-o', '--outdir', default=None, help="Output directory.")
    p.add_argument('--force', action='store_true', help="Run even if output "
                        "directory already exists (pre-existing runs will not be "
                        "repeated).")
    p.add_argument('-s', '--submit', action='store_true', help="Submit slurm "
                        "job automatically.")
    p.add_argument('--ntasks', default=-1, type=int,
                        help="Maximum number of tasks request through SLURM.")
    p.add_argument('--device-count', type=int, default=None,
                        help="Number of GPUs/CPUs to use for each task "
                        "(default: 1 for GPUs and 4 for CPUs)")
    p.add_argument('--platform', choices=['cpu', 'gpu'], default='cpu',
                        help="Platform to run on (default 'cpu').")
    p.add_argument('--omp-num-threads', type=int, default=1,
                        help="Number of threads for numpy.")
    p.add_argument('-C', '--constraints', help="SLURM constraints.")
    p.add_argument('-t', '--time', help="SLURM time directive.")
    p.add_argument('--seed', default=None, type=int, help="Random seed.")
    p.add_argument('-v', '--verbose', action='store_true')
    return p

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    # set up logging and procure Git repo hash
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # load config file
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"unable to load: {config_path}")
    logging.info(f"Loading config from {config_path}")
    config = rd.utils.load_config(config_path)

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
        'run_config': 'engine/{t0:.6f}/config.ini',
        'run_result': 'engine/{t0:.6f}/result.nc',
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
        f.write("\n# ringdown v {}".format(__version__))

    # Identify target analysis times. There will be three possibilities:
    #     1- listing the times explicitly
    #     2- listing time differences with respect to a reference time
    #     3- providing start, stop, step instructions to construct start times
    #        (potentially relative to a reference time)
    # Time steps/differences can be specified in seconds or M, if a reference mass
    # is provided (in solar masses).

    t0s = TargetCollection.from_config(config).get('t0')

    # Record start times
    np.savetxt(PATHS["t0"], t0s)

    ##############################################################################
    # SET UP RUNS
    ##############################################################################

    if rerun and os.path.exists(PATHS['run_task']):
        os.remove(PATHS['run_task'])

    # determine how many devices to use, will default to 1 for GPUs and 4 for CPUs
    if args.device_count is not None:
        NDEVICE = args.device_count
    elif args.platform == 'gpu':
        NDEVICE = 1
    else:
        NDEVICE = 4

    task_opts = [
        "-o {result}",
        f"--platform {args.platform}",
        f"--device-count {NDEVICE}",
        f"--omp-num-threads {args.omp_num_threads}",
        "--verbose"
    ]

    TASK = "mkdir -p {rundir}; cd {rundir}; ringdown_fit {config} %s"\
        "&>> run.log\n" % ' '.join(task_opts)

    logging.info(f"Processing {len(t0s)} start times (recorded in {PATHS['t0']})")
    for i, t0 in enumerate(t0s):
        # Set up child configuration file
        cpath = PATHS['run_config'].format(i=i, t0=t0)
        rundir = os.path.dirname(cpath)
        os.makedirs(rundir, exist_ok=True)
        config_child = configparser.ConfigParser()
        config_child.read(PATHS['config'])

        # Set start time
        # Restore conditioning section
        config_child['target']['t0'] = str(t0)

        # Write out config file for this run
        rpath = PATHS['run_result'].format(i=i, t0=t0)
        if os.path.exists(rpath):
            logging.info(f"Run {i} already exists. Will skip.")
        else:
            with open(cpath, 'w') as f:
                config_child.write(f)
            with open(PATHS['run_task'], 'a') as f:
                f.write(TASK.format(rundir=rundir, result=rpath, config=cpath))

    logging.info("Done processing {} runs.".format(len(t0s)))


    ##############################################################################
    # SLURM WORKFLOW
    ##############################################################################

    # If given args.ntasks, then request that many tasks; otherwise try to have as
    # many tasks as there are runs to process.
    if args.ntasks < 0:
        NTASK = len(t0s)
    else:
        NTASK = args.ntasks

    # a user may allocate up to 1280 cores at one time in a center partition, so
    # issue a warning if this limit will be exceeded
    if NTASK*NDEVICE > 1280:
        w = "Requested number of cores ({}) above 1280 user limit."
        logging.warning(w.format(NTASK*NDEVICE))

    if args.platform == 'cpu':
        EXE = [
            '#! /usr/bin/env bash',
            'cd {}'.format(outdir),
            f"sbatch -p cca -n {NTASK} -c {NDEVICE} disBatch {PATHS['run_task']}",
            'cd -',
        ]
    else:
        # see https://wiki.flatironinstitute.org/SCC/Software/UsingTheGPUNodes
        NCPU = 16
        EXE = [
            '#! /usr/bin/env bash',
            'cd {}'.format(outdir),
            f"sbatch -p gpu -n {NTASK} --gpus-per-task={NDEVICE} --cpus-per-task={NCPU} --gpu-bind=closest disBatch {PATHS['run_task']}",
            'cd -',
        ]

    if args.constraints:
        EXE[2] = EXE[2].replace('sbatch', f"sbatch -C {args.constraints}")

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

if __name__ == '__main__':
    main()