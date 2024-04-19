import numpy as np

def filter(rdz, rate=0.1, **kwargs):
    """
    Randomly mask rdz rows
    """
    assert (rate>0) and (rate<1)
    mask = np.random.choice([True, False],p=[rate, 1-rate], size=rdz.shape[0])
    filtered_rdz = rdz[mask]
    weight = np.ones(filtered_rdz.shape[0])
    return filtered_rdz, weight