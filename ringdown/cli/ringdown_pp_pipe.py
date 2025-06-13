#!/usr/bin/env python
# coding: utf-8

# Copyright 2025
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
import ringdown as rd
import argparse
import logging
import time
from ringdown import __version__
from ringdown.config import PIPE_SECTION


##############################################################################
# PARSE INPUT
##############################################################################


def get_parser():
    p = argparse.ArgumentParser(description="Ringdown PP analysis")
    p.add_argument('config', help="Path to configuration file.")
    p.add_argument('-o', '--outdir', default=None, help="Output directory.")
    p.add_argument('--force', action='store_true', help="Run even if output "
                   "directory already exists (pre-existing runs will not be "
                   "repeated).")
    p.add_argument('-s', '--submit', action='store_true', help="Submit slurm "
                   "job automatically.")
    p.add_argument('--ntasks', default=-1, type=int,
                   help="Maximum number of tasks to request through SLURM.")
    p.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Platform to run on (default 'cpu').",
    )
    p.add_argument("-C", "--constraints", help="SLURM constraints.")
    p.add_argument("-t", "--time", help="SLURM time directive.")
    p.add_argument('--seed', default=None, type=int, help="Random seed.")
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    # set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.info(f"ringdown_pp_pipe v{__version__}")

    # load config file
    config = rd.utils.load_config(os.path.abspath(args.config))

    # set random seed (purposedly fail if not provided)
    seed = args.seed or config.getint(PIPE_SECTION, 'seed')
    logging.info(f"Random seed set to {seed}")
    rng = np.random.default_rng(seed)

    # determine run directory
    outdir = args.outdir or config.get(PIPE_SECTION, 'outdir', fallback=None)
    if not outdir:
        outdir = f'ringdown_pp_{time.time():.0f}'
        logging.warning(f"No run directory provided, defaulting to {outdir}")
    outdir = os.path.abspath(outdir)
    logging.info(f"Running in {outdir}")

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
        'acf': 'acf{ifo}.dat',
        'prior': 'prior.nc',
        'prior_exe': 'prior.py',
        'run_config': 'engine/pp_{i}/config.ini',
        'run_result': 'engine/pp_{i}/result.nc',
        'run_task': 'TaskFile',
        'exe': 'submit.sh',
    }
    PATHS = {k: os.path.join(outdir, v) for k, v in PATHS.items()}

    # record config and arguments for reproducibility
    with open(PATHS['config'], 'w') as f:
        config.write(f)

    with open(PATHS['command'], 'w') as f:
        f.write(f"{' '.join(sys.argv)}\n\n")
        for k, v in vars(args).items():
            f.write(f"# {k}: {v}\n")
        f.write(f"\n# {os.path.realpath(__file__)}")
        f.write(f"\n# ringdown v {__version__}")

    ##############################################################################
    # SET UP RUNS
    ##############################################################################

    # Get number of injections
    nruns = config.getint(PIPE_SECTION, 'nruns')

    # Create a fit object, which we will use to manipulate data and obtain prior
    # this fit will automatically take care of all data handling based on config
    fit = rd.Fit.from_config(config)

    # cache ACFs
    for ifo, acf in fit.acfs.items():
        acf.to_csv(PATHS['acf'].format(ifo=ifo), sep='\t', header=None)

    # ----------------------------------------------------------------------------
    # Select analysis times

    # Determine boundaries of injection region: an interval determined by
    # t0+/-inj_zone_duration, with an exclusion zone carved outaround
    # the indicated trigger time t0 (potentially of zero width, if no exclusion
    # required, i.e., `safe_zone_duration = 0`); by default, the injection region
    # is the entire time range of the data.
    safe_zone_duration = config.getfloat(PIPE_SECTION, 'safe-zone-duration',
                                         fallback=0)
    inj_zone_duration = config.getfloat(PIPE_SECTION, 'inj-zone-duration',
                                        fallback=np.inf)

    # Pick as many random times uniformly within the injection region as the
    # requested number of injections. Do this in two steps: first, select randomly
    # from before and after the censored trigger time; then, choose randomly
    # between before and after for each entry.
    #
    # NOTE: this selects different times for each detector, but only the ones for
    # the first detector will be used unless the `timeslides` option is given.
    t_chosen = {}
    for ifo, data in fit.data.items():
        t = data.time
        t_min = max(fit.t0 - inj_zone_duration, min(t))
        t_max = min(fit.t0 + inj_zone_duration, max(t))

        t_before = rng.uniform(t_min, fit.t0 - safe_zone_duration, nruns)
        t_after = rng.uniform(fit.t0 + safe_zone_duration, t_max, nruns)
        mask = rng.integers(0, 2, nruns)
        t_chosen[ifo] = mask*t_before + (1 - mask)*t_after

    # ----------------------------------------------------------------------------
    # Set up children fits

    if rerun and os.path.exists(PATHS['run_task']):
        os.remove(PATHS['run_task'])

    # determine how many devices to use, will default to 1 for GPUs and
    # 4 for CPUs
    if "RINGDOWN_DEVICE_COUNT" in os.environ:
        NDEVICE = int(os.environ["RINGDOWN_DEVICE_COUNT"])
    elif args.platform == "gpu":
        NDEVICE = 1
    else:
        NDEVICE = 4

    # Set the environment variable so child processes inherit it
    os.environ["RINGDOWN_DEVICE_COUNT"] = str(NDEVICE)

    task_opts = [
        "-o {result}",
        f"--platform {args.platform}",
        "--verbose",
    ]

    TASK = (
        "mkdir -p {rundir}; cd {rundir}; ringdown_fit {config} %s"
        "&>> run.log\n" % " ".join(task_opts)
    )

    logging.info("Ready to process {} runs.".format(nruns))
    for i in range(nruns):
        # Set up child configuration file
        cpath = PATHS['run_config'].format(i=i)
        # TODO: fix for caching
        rundir = os.path.dirname(cpath)
        os.makedirs(rundir, exist_ok=True)
        config_child = rd.utils.load_config(PATHS['config'])

        # Point to data
        # TODO: modify if allowing on-the-fly noise generation
        config_child['data']['path'] = config['data']['path']

        # Point to the ACFs we cached above
        config_child['acf'] = dict(path=PATHS['acf'], float_precision='round_trip',
                                   sep='\\t', header='None')

        # By default, set the target time to be the one corresponding to the first
        # detector in the random selection made above
        t0 = t_chosen[fit.ifos[0]][i]
        config_child['target']['t0'] = str(t0)
        # If `timeslides` is turned on, then pick random times for all
        # detectors (as chosen above), irrespective of the Target sky location.
        if config[PIPE_SECTION].getboolean('timeslides', False):
            delays = {ifo: t[i] - t0 for ifo, t in t_chosen.items()}
            config_child['target']['slide'] = str({i: float(v)
                                                   for i, v in delays.items()})

        # Write out config file for this run
        rpath = PATHS['run_result'].format(i=i)
        if os.path.exists(rpath):
            logging.info("Run {} already exists. Will skip.".format(i))
        else:
            with open(cpath, 'w') as f:
                config_child.write(f)
            with open(PATHS['run_task'], 'a') as f:
                f.write(TASK.format(rundir=rundir, result=rpath, config=cpath))

    logging.info("Done processing {} runs.".format(nruns))

    ##############################################################################
    # SLURM WORKFLOW
    ##############################################################################

    PRIOR_EXE = [
        "#!/usr/bin/env python",
        "",
        "import multiprocessing",
        "multiprocessing.set_start_method('fork')",
        "import numpy as np",
        "from glob import glob",
        "import ringdown as rd",
        "import arviz as az",
        "import os",
        "",
        "if os.path.exists('{}'):".format(PATHS['prior']),
        "    prior = az.from_netcdf('{}')".format(PATHS['prior']),
        "else:",
        "    fit = rd.Fit.from_config('{}')".format(PATHS['config']),
        "    fit.run(prior=True)",
        "    fit.prior.to_netcdf('{}')".format(PATHS['prior']),
        "    prior = fit.prior",
        "",
        "# initialize RNG",
        "rng = np.random.default_rng({})".format(seed),
        "",
        "# stack samples",
        "data = prior.posterior.stack(sample=('chain', 'draw'))",
        "# reoder samples randomly",
        "random_subset = rng.permutation(np.arange(len(data['sample'])))",
        "data = data.isel(sample=random_subset)",
        "",
        "configs = sorted(glob('{}'))".format(
            PATHS['run_config'].format(i='*')),
        "for i,p in enumerate(configs):",
        "    if os.path.exists('{}'.format(i=i)):".format(PATHS['run_result']),
        "        print('WARNING: skipping run {}'.format(i))",
        "    else:",
        "        with open(p, 'r+') as f:",
        "            if '[injection]' in f.read():",
        "                print('WARNING: skipping run {} (injection already in config)'.format(i))",
        "            else:",
        "                f.write('\\n[injection]\\n')",
        "                f.write('# sample {}\\n'.format(i))",
        "                f.write(\"# {}\\n\")".format(fit),
        "                for k, v in data.isel(sample=i).data_vars.items():",
        "                    if 'h_det' not in k:",
        "                        v_str = np.array2string(v.values, separator=', ')",
        "                        f.write('{} = {}\\n'.format(k, v_str))",
    ]

    epath = PATHS['prior_exe']
    with open(epath, 'w') as f:
        f.write('\n'.join(PRIOR_EXE))
    st = os.stat(epath)
    os.chmod(epath, st.st_mode | 0o111)
    logging.info("Wrote prior executable: {}".format(epath))

    # ----------------------------------------------------------------------------
    # Set up slurm workflow

    # If given args.ntasks, then request that many tasks; otherwise try to have
    # as many tasks as there are runs to process.
    if args.ntasks < 0:
        NTASK = nruns
    else:
        NTASK = args.ntasks

    # a user may allocate up to 1280 cores at one time in a center partition, so
    # issue a warning if this limit will be exceeded
    if NTASK * NDEVICE > 1280:
        w = "Requested number of cores ({}) above 1280 user limit."
        logging.warning(w.format(NTASK * NDEVICE))

    # check for slurm constraints
    if args.constraints:
        command = f"sbatch -C {args.constraints} "
    else:
        command = "sbatch"

    if args.platform == "cpu":
        EXE = [
            '#! /usr/bin/env bash',
            '',
            'function get_id() {',
            '    if [[ "$1" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then',
            '        echo "${BASH_REMATCH[1]}"',
            '        exit 0',
            '    else',
            '        echo "sbatch failed"',
            '        exit 1',
            '    fi',
            '}',
            '',
            f'cd {outdir}',
            '# prior job',
            f'priorid=$(get_id "$(sbatch -p cca -c 4 -t 0-1 {PATHS["prior_exe"]})")',
            '',
            '# pp jobs',
            f'{command} --dependency=afterok:$priorid -p cca -n {NTASK} -c {NDEVICE} disBatch {PATHS["run_task"]}',
            'cd -',
        ]
    else:
        # these options are set to match the GPU nodes at the Flatiron Institute
        # see https://wiki.flatironinstitute.org/SCC/Software/UsingTheGPUNodes
        NCPU = 16
        prior_command = f"sbatch -p gpu --gpus-per-task=1 --cpus-per-task={NCPU} {PATHS['prior_exe']}"
        pp_command = (
            f"{command} --dependency=afterok:$priorid -p gpu -n {NTASK} "
            f"--gpus-per-task={NDEVICE} "
            f"--cpus-per-task={NCPU} "
            f"--gpu-bind=closest disBatch {PATHS['run_task']}"
        )

        EXE = [
            '#! /usr/bin/env bash',
            '',
            'function get_id() {',
            '    if [[ "$1" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then',
            '        echo "${BASH_REMATCH[1]}"',
            '        exit 0',
            '    else',
            '        echo "sbatch failed"',
            '        exit 1',
            '    fi',
            '}',
            '',
            f'cd {outdir}',
            '# prior job',
            f'priorid=$(get_id "$({prior_command})")',
            '',
            '# pp jobs',
            pp_command,
            'cd -',
        ]

    epath = PATHS['exe']
    with open(epath, 'w') as f:
        f.write('\n'.join(EXE))
    st = os.stat(epath)
    os.chmod(epath, st.st_mode | 0o111)
    if args.submit:
        print(f"Submitting: {epath}")
        import subprocess
        subprocess.run(epath)
    else:
        print(f"Submit by running: {epath}")


if __name__ == "__main__":
    main()
