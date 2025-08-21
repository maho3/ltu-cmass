import numpy as np

"""
Hardcoded, HOD priors for CMASS observational samples

Each prior is Gaussian with a mean and stdev defined over three interpolated 
redshift bins.

Computed in notebooks/constrain_hod_prior.ipynb
"""

SURVEY_HOD_PRIORS = {
    # zpivot = [0.40, 0.50, 0.70]  (edges and hump)
    '0.40,0.50,0.70': {
        'simbig': {
            'mean': [13.147074, 12.587817, 13.521771],
            'stdev': [0.3698568, 0.33010304, 0.27506068],
        },
        'sgc': {
            'mean': [13.614453, 12.449558, 13.586691],
            'stdev': [0.20270658, 0.2578436, 0.17996997],
        },
        'mtng': {
            'mean': [12.5065565, 12.930511, 13.391156],
            'stdev': [0.31901833, 0.2493872, 0.21629056],
        },
        'ngc': {
            'mean': [13.640792, 12.395513, 13.521472],
            'stdev': [0.23169038, 0.2704503, 0.1973685],
        }
    },
    # zpivot = [0.45, 0.55, 0.65]  (evenly-spaced)
    '0.45,0.55,0.65': {
        'simbig': {
            'mean': [12.652918, 12.548699, 13.528817],
            'stdev': [0.30988806, 0.2718738, 0.23241553],
        },
        'sgc': {
            'mean': [12.745239, 12.441549, 13.321042],
            'stdev': [0.29355767, 0.20268019, 0.18492053],
        },
        'mtng': {
            'mean': [12.633412, 12.996082, 13.222733],
            'stdev': [0.29536995, 0.23001276, 0.1950261],
        },
        'ngc': {
            'mean': [12.835313, 12.455541, 13.390458],
            'stdev': [0.2084813, 0.19274066, 0.1532904],
        }
    }
}

# for survey in SURVEY_HOD_PRIORS.values():
#     survey['mean'] = np.array(survey['mean'])
#     survey['stdev'] = np.array(survey['stdev'])
