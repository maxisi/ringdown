#!/usr/bin/env python
# coding: utf-8
#
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

import os
import argparse
from ast import literal_eval
import logging
import ringdown as rd

##############################################################################
# PARSE INPUT
##############################################################################

DEFOUT = "ringdown_fit.nc"
_HELP = "Set up and run a ringdown analysis from a configuration file."


def get_parser():
    p = argparse.ArgumentParser(description=_HELP)
    p.add_argument('config', help="path to configuration file.")
    p.add_argument('-o', '--output', default=None,
                   help="output result path (default: `{}`).".format(DEFOUT))
    p.add_argument('--prior', action='store_true', help="sample from prior.")
    p.add_argument('--omp-num-threads', help='number of threads for numpy.',
                   type=int, default=1)
    p.add_argument('--platform', choices=['cpu', 'gpu'], default='cpu',
                   help="device platform (default: cpu).")
    p.add_argument('--device-count', type=int, default=4,
                   help="number of devices to use.")
    p.add_argument('--force', action='store_true',
                   help="overwrites output file if it already exists.")
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(args=None, defout=DEFOUT):
    parser = get_parser()
    args = parser.parse_args(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cpu_count = os.cpu_count()

    # check for numpy threading options
    if args.omp_num_threads > cpu_count:
        logging.warning(f"requested OMP_NUM_THREADS ({args.omp_num_threads}) "
                        "greater than the number of available CPUs. Setting "
                        f"it to the maximum number of CPUs ({cpu_count}).")
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    else:
        logging.info("OMP_NUM_THREADS set to: {}".format(args.omp_num_threads))
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    print("Loading: {}".format(os.path.abspath(args.config)))

    config = rd.utils.load_config(args.config)

    if config.has_section('run'):
        run_kws = {k: literal_eval(v) for k, v in config['run'].items()}
        if run_kws.pop('omp_num_threads', False):
            logging.warning("omp_num_threads is set in the configuration file,"
                            " but it will be ignored. Use the command line "
                            "option instead.")
    else:
        run_kws = {}
    run_kws['prior'] = args.prior or run_kws.get('prior', False)

    from jax import config as jax_config
    jax_config.update("jax_enable_x64", not run_kws.pop('float32', False))

    if run_kws['prior']:
        defout = defout.replace('fit', 'prior')
    out = args.output or defout

    import numpyro
    numpyro.set_platform(args.platform)
    if args.device_count is not None:
        if args.platform == 'cpu' and args.device_count > cpu_count:
            logging.warning(f"requested device count ({args.device_count}) "
                            "greater than the number of available CPUs. "
                            "Setting it to the maximum number of CPUs "
                            f"({cpu_count}).")
            args.device_count = cpu_count
        else:
            numpyro.set_host_device_count(args.device_count)

    if os.path.exists(out):
        if args.force:
            logging.warning("overwriting output file: {}".format(out))
        else:
            raise FileExistsError("output file already exists: {}".format(out))

    ##########################################################################
    # RUN FIT
    ##########################################################################

    fit = rd.Fit.from_config(config)
    fit.run(**run_kws)

    if run_kws['prior']:
        result = fit.prior
    else:
        result = fit.result

    ext = os.path.splitext(out)[-1]
    if ext.lower() == '.nc':
        result.to_netcdf(out)
    else:
        result.to_json(out)

    print("Saved ringdown fit: {}".format(out))

if __name__ == '__main__':
    main()