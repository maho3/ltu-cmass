# Postprocess saved outputs of FastPM simulation

import hydra
import logging
import os
from omegaconf import DictConfig, OmegaConf

from ..utils import get_source_path, timing_decorator, save_cfg
from .tools import parse_nbody_config
from cmass.nbody.fastpm import process_outputs


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody', 'multisnapshot'])

    # Build run config
    cfg = parse_nbody_config(cfg)
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

    # Locate the (transient) work directory holding the FastPM snapshots.
    # NOTE: with node-local /tmp staging the snapshots do not survive the job,
    # so standalone postprocessing only works when meta.scratchdir is unset
    # (or equals wdir), i.e. snapshots persist alongside outdir.
    scratch_base = cfg.meta.get('scratchdir', None) or cfg.meta.wdir
    workdir = get_source_path(
        scratch_base, cfg.nbody.suite, "fastpm",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )

    rho, fvel, pos, vel = process_outputs(
        cfg, workdir, outdir, delete_files=True)

    logging.info("Done!")


if __name__ == '__main__':
    main()
