# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
The entry point for training and inference weathergen-atmo
"""

import logging
import os
import pdb
import sys
import time
import traceback
from pathlib import Path

import torch
import weathergen.common.config as config
from omegaconf import OmegaConf
from weathergen.common.logger import init_loggers

import weathergen.utils.cli as cli
from weathergen.train.trainer import Trainer

logger = logging.getLogger(__name__)


def train() -> None:
    """Entry point for calling the training code from the command line."""
    main([cli.Stage.train] + sys.argv[1:])


def train_continue() -> None:
    """Entry point for calling train_continue from the command line."""
    main([cli.Stage.train_continue] + sys.argv[1:])


def inference():
    """Entry point for calling the inference code from the command line."""
    main([cli.Stage.inference] + sys.argv[1:])


def main(argl: list[str]):
    try:
        argl = _fix_argl(argl)
    except ValueError as e:
        logger.error(str(e))

    parser = cli.get_main_parser()
    args = parser.parse_args(argl)
    match args.stage:
        case cli.Stage.train:
            run_train(args)
        case cli.Stage.train_continue:
            run_continue(args)
        case cli.Stage.inference:
            run_inference(args)
        case _:
            logger.error("No stage was found.")


def _fix_argl(argl):  # TODO remove this fix after grace period
    """Ensure `stage` positional argument is in arglist."""
    if argl[0] not in cli.Stage:
        try:
            stage = os.environ.get("WEATHERGEN_STAGE")
        except KeyError as e:
            msg = (
                "`stage` postional argument and environment variable 'WEATHERGEN_STAGE' missing.",
                "Provide either one or the other.",
            )
            raise ValueError(msg) from e

        argl = [stage] + argl

    return argl


def run_inference(args):
    """
    Inference function for WeatherGenerator model.

    Note: Additional configuration for inference (`test_config`) is set in the function.
    """

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    # Seed inference too. Without this the torch RNG differs between runs, and since
    # sampling and, until eval() is set, dropout both draw from it, the same rollout
    # configuration produces a different trajectory every time.
    seed = cf.get("data_loading", {}).get("rng_seed", None)
    if seed is not None and not OmegaConf.is_missing(cf.data_loading, "rng_seed"):
        torch.manual_seed(int(seed))
        logger.info("inference rng_seed = %d", int(seed))

    devices = Trainer.init_torch()
    cf = Trainer.init_ddp(cf)

    init_loggers(log_path=config.get_path_logs(cf))

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = Trainer(cf.train_logging)
    try:
        trainer.inference(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


def run_continue(args):
    """
    Function to continue training for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)

    mp_method = cf.general.get("multiprocessing_method", "fork")
    devices = Trainer.init_torch(multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)

    init_loggers(log_path=config.get_path_logs(cf))

    # track history of run to ensure traceability of results
    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = Trainer(cf.train_logging)

    try:
        trainer.run(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


def run_train(args):
    """
    Training function for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

    cli_overwrite = config.from_cli_arglist(args.options)

    cf = config.load_merge_configs(
        args.private_config, None, None, args.base_config, *args.config, cli_overwrite
    )
    cf = config.set_run_id(cf, args.run_id, False)

    # Only auto-seed when the user has not supplied one. Unconditionally assigning
    # here silently overrides `--options data_loading.rng_seed=N`, which makes runs
    # irreproducible and makes seed-controlled A/B comparisons impossible.
    if OmegaConf.is_missing(cf.data_loading, "rng_seed") or cf.data_loading.rng_seed is None:
        cf.data_loading.rng_seed = int(time.time())
        _seed_source = "auto (wall clock)"
    else:
        _seed_source = "user-specified"
    # identical on every rank: nothing broadcasts rank 0's initial weights
    torch.manual_seed(cf.data_loading.rng_seed)
    logger.info("rng_seed = %d (%s)", cf.data_loading.rng_seed, _seed_source)
    mp_method = cf.general.get("multiprocessing_method", "fork")
    devices = Trainer.init_torch(multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)

    # this line should probably come after the processes have been sorted out else we get lots
    # of duplication due to multiple process in the multiGPU case
    init_loggers(log_path=config.get_path_logs(cf))

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.streams = config.load_streams(Path(cf.streams_directory))

    if cf.with_flash_attention:
        assert cf.with_mixed_precision

    trainer = Trainer(cf.train_logging)

    try:
        trainer.run(cf, devices)
    except Exception:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        if cf.world_size == 1:
            pdb.post_mortem(tb)


if __name__ == "__main__":
    main(sys.argv[1:])
