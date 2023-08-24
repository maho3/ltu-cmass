import sys
import logging
import yaml
import datetime
import os
from os.path import join as pjoin
import numpy as np
import json


class attrdict(dict):
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


def setup_logger(logdir, name='log', level=logging.INFO):
    # define a naming prefix
    date = datetime.datetime.now().strftime("%Y%m%d")
    prefix = f'{name}_{date}'

    # find all logs with the same prefix
    logs = os.listdir(logdir)
    logs = [l for l in logs if l.startswith(prefix)]

    # find the next available number
    path_to_log = pjoin(logdir, f'{name}_{date}_{len(logs):04}.log')

    # setup the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s-%(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(path_to_log),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging to {path_to_log}")


def get_global_config():
    with open('global.cfg', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def timing_decorator(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}...")
        t0 = datetime.datetime.now()
        out = func(*args, **kwargs)
        dt = (datetime.datetime.now() - t0).total_seconds()
        logging.info(
            f"Done running {func.__name__}. "
            f"Time elapsed: {int(dt//60)}m{int(dt%60)}s.")
        return out
    return wrapper


@timing_decorator
def load_params(index, cosmofile):
    if index == "fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open(cosmofile, 'r') as f:
        content = f.readlines()[index+1]
    content = [np.float64(x) for x in content.split()]
    return content
