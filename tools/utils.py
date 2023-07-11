import sys
import logging
import yaml
import datetime
from os.path import join as pjoin


def get_logger(logdir, level=logging.INFO):
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    date = date[:-4]  # remove last 4 digits of microseconds
    path_to_log = pjoin(logdir, f'log{date}.log')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s-%(name)s-%(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(path_to_log),
            logging.StreamHandler(sys.stdout)
        ]
    )


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
