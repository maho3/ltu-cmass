import numpy as np
from os.path import join
import hydra
import logging
import os
import bigfile
from omegaconf import DictConfig, OmegaConf
import shutil
import subprocess
import h5py
import multiprocessing as mp

from ..utils import get_source_path, timing_decorator, save_cfg
from .tools import (
    parse_nbody_config, get_ICs,
    save_white_noise_grafic, generate_pk_file, rho_and_vfield,
    save_nbody
)
from cmass.nbody.fastpm import process_outputs

@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody'])

    # Build run config
    cfg = parse_nbody_config(cfg, lightcone=True)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Create output directory
    outdir = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, "fastpm",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )

    rho, fvel, pos, vel = process_outputs(cfg, outdir, delete_files=True)

    logging.info("Done!")


if __name__ == '__main__':
    main()