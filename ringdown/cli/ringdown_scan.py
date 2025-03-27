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
from jax import config as jax_config
import numpyro
import ringdown as rd
from ringdown.config import PIPE_SECTION

##############################################################################
# PARSE INPUT
##############################################################################

DEFOUT = "ringdown_scan/*.nc"
_HELP = "Set up and run a ringdown analysis from a configuration file."


def get_parser():
    p = argparse.ArgumentParser(description=_HELP)
    p.add_argument("config", help="path to configuration file.")
    p.add_argument(
        "-o",
        "--output",
        default=DEFOUT,
        help=f"output result path (default: `{DEFOUT}`).",
    )
    p.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="cpu",
        help="device platform (default: cpu).",
    )
    p.add_argument(
        "--device-count", type=int, default=4, help="number of devices to use."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrites output file if it already exists.",
    )
    p.add_argument(
        "--individual-progress-bars",
        action="store_true",
        help="show progress bar for each target.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cpu_count = os.cpu_count()

    # check for numpy threading options
    if "OMP_NUM_THREADS" not in os.environ:
        logging.info("Setting OMP_NUM_THREADS to 1.")
        os.environ["OMP_NUM_THREADS"] = "1"

    print(f"Loading: {os.path.abspath(args.config)}")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"config file not found: {args.config}")

    config = rd.utils.load_config(args.config)

    if config.has_section("run"):
        run_kws = {k: literal_eval(v) for k, v in config["run"].items()}
    else:
        run_kws = {}
    run_kws["individual_progress_bars"] = (
        args.individual_progress_bars or args.verbose
    )

    jax_config.update("jax_enable_x64", not run_kws.pop("float32", False))

    out = os.path.abspath(args.output or DEFOUT)
    out = config.get(PIPE_SECTION, "outpath", fallback=out)

    numpyro.set_platform(args.platform)
    if args.device_count is not None:
        if args.platform == "cpu" and args.device_count > cpu_count:
            logging.warning(
                f"requested device count ({args.device_count}) "
                "greater than the number of available CPUs. "
                "Setting it to the maximum number of CPUs "
                f"({cpu_count})."
            )
            args.device_count = cpu_count
        else:
            numpyro.set_host_device_count(args.device_count)

    ##########################################################################
    # RUN FIT
    ##########################################################################

    fit = rd.FitSequence.from_config(config)

    # check if output files exist
    if not args.force:
        new_targets = []
        for t0, target in fit.targets:
            path = fit.format_output_path(out, t0)
            if os.path.exists(path):
                logging.warning(
                    f"output file already exists (skipping target): {path}"
                )
            else:
                outdir = os.path.dirname(path)
                os.makedirs(outdir, exist_ok=True)
                new_targets.append(target)
        if not new_targets:
            logging.warning(
                "all output files already exist: exiting "
                "(use --force to overwrite)."
            )
            return
        fit.set_target_collection(new_targets)

    fit.run(**run_kws, output_path=out)

    print(f"Saved ringdown fits: {out}")


if __name__ == "__main__":
    main()
