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
    p.add_argument("config", help="path to configuration file.")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="output result path (default: `{}`).".format(DEFOUT),
    )
    p.add_argument("--prior", action="store_true", help="sample from prior.")
    p.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="cpu",
        help="device platform (default: cpu).",
    )
    p.add_argument(
        "--device-count",
        type=int,
        default=None,
        help="number of CPU host devices for parallel chains; CPU only "
             "(default: RINGDOWN_DEVICE_COUNT environment variable, or 4; "
             "clamped to the number of available CPUs). On GPU/TPU the "
             "device count is not controlled here (visible devices are set "
             "through the environment, e.g., CUDA_VISIBLE_DEVICES) and an "
             "explicit value is ignored with a warning.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrites output file if it already exists.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(args=None, defout=DEFOUT):
    parser = get_parser()
    args = parser.parse_args(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    print(f"Loading: {os.path.abspath(args.config)}")

    config = rd.utils.load_config(args.config)

    if config.has_section("run"):
        run_kws = {k: literal_eval(v) for k, v in config["run"].items()}
    else:
        run_kws = {}
    run_kws["prior"] = args.prior or run_kws.get("prior", False)

    rd.setup(
        platform=args.platform,
        num_devices=args.device_count,
        x64=not run_kws.pop("float32", False),
    )

    if run_kws["prior"]:
        defout = defout.replace("fit", "prior")
    out = args.output or defout

    if os.path.exists(out):
        if args.force:
            logging.warning(f"overwriting output file: {out}")
        else:
            raise FileExistsError(f"output file already exists: {out}")

    ##########################################################################
    # RUN FIT
    ##########################################################################

    fit = rd.Fit.from_config(config)
    fit.run(**run_kws)

    if run_kws["prior"]:
        result = fit.prior
    else:
        result = fit.result

    ext = os.path.splitext(out)[-1]
    if ext.lower() != ".nc":
        logging.warning(
            f"unsupported output format {ext!r}: only netCDF (.nc) output is supported")
        out = out + ".nc"

    result.to_netcdf(out)

    print(f"Saved ringdown fit: {out}")


if __name__ == "__main__":
    main()
