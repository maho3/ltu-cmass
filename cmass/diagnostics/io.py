"""This module provides functions to load and save diagnostic data."""


import logging
import h5py
import datetime

from ..utils import save_configuration_h5


def save_group(file, data, attrs=None, a=None, config=None, save_HOD=False):
    logging.info(f'Saving {len(data)} datasets to {file}')
    with h5py.File(file, 'a') as f:
        if a is not None:
            group = f.require_group(a)
        else:
            group = f
        if attrs is not None:
            for key, value in attrs.items():
                group.attrs[key] = value
            group.attrs['timestamp'] = datetime.datetime.now().isoformat()
        for key, value in data.items():
            if key in group:
                del group[key]
            group.create_dataset(key, data=value)

        if config is not None:
            save_configuration_h5(f, config, save_HOD=save_HOD)
