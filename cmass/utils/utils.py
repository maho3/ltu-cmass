import sys
import logging
import yaml
import datetime
import os
from os.path import join as pjoin
import json


class attrdict(dict):  # TODO: remove?
    """Simple dict wrapper, allowing attribute access and saving to yaml."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self, f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(json.load(f))


def get_global_config():
    with open('global.cfg', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_source_path(wdir, simtype, L, N, lhid, check=True):
    # get the path to the source directory, and check at each level
    sim_dir = pjoin(wdir, simtype)
    cfg_dir = pjoin(sim_dir, f'L{L}-N{N}')
    lh_dir = pjoin(cfg_dir, str(lhid))

    if check:
        if not os.path.isdir(sim_dir):
            raise ValueError(
                f"Simulation directory {sim_dir} does not exist.")
        if not os.path.isdir(cfg_dir):
            raise ValueError(
                f"Configuration directory {cfg_dir} does not exist.")
        if not os.path.isdir(lh_dir):
            raise ValueError(
                f"Latin hypercube directory {lh_dir} does not exist.")
    return lh_dir


def timing_decorator(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}...")
        t0 = datetime.datetime.now()
        out = func(*args, **kwargs)
        dt = (datetime.datetime.now() - t0).total_seconds()
        logging.info(
            f"Finished {func.__name__}... "
            f"({int(dt//60)}m{int(dt%60)}s)")
        return out
    return wrapper


@timing_decorator
def load_params(index, cosmofile):
    if index == "fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open(cosmofile, 'r') as f:
        content = f.readlines()[index]
    content = [float(x) for x in content.split()]
    return content
