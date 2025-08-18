import numpy as np

"""
Hardcoded, standardized survey geometries for CMASS.

The 'rotation' matrix is first applied to align the median line-of-sight (LOS) 
with the x-axis and the largest variance along the y-axis. After rotation, the
box is positioned with size 'boxsize' centered at 'boxcenter'.

Computed in notebooks/fit_box_geometry.ipynb
"""

SURVEY_GEOMETRIES = {
    'simbig': {
        'fsky': 0.0482,
        'boxsize': 1656.2,
        'boxcenter': [1284.40507414,    4.43608373,  -31.45720169],
        'rotation': [[0.96868018,  0.04538794,  0.24412834],
                     [0.02920341, -0.99715353,  0.06951251],
                     [0.24658847, -0.06020601, -0.96724835]],  # TODO: fill in actual value
    },
    'sgc': {
        'fsky': 0.0688,
        'boxsize': 2929.7,
        'boxcenter': [1248.30163954,    4.85321152,  -33.61562497],
        'rotation': [[0.97755752,  0.04542538,  0.20571297],
                     [0.02316036, -0.99373062,  0.10937571],
                     [0.20939171, -0.10215666, -0.97248091]],
    },
    'mtng': {
        'fsky': 0.1257,
        'boxsize':  2945.3,
        'boxcenter': [1187.69330064,   54.20128635,  194.84908638],
        'rotation': [[0.57691293,  0.57758749,  0.57755014],
                     [-0.06669677, -0.67141109,  0.7380777],
                     [0.81407802, -0.46432729, -0.34882252]],
    },
    'ngc': {
        'fsky': 0.1822,
        'boxsize': 3750.0,
        'boxcenter': [1063.28224848,  -12.57940443,  -11.66296504],
        'rotation': [[-0.82592816, -0.0703174,  0.55937298],
                     [0.09292135, -0.99560058,  0.0120462],
                     [0.55606501,  0.06192699,  0.82882854]],
    }
}

for survey in SURVEY_GEOMETRIES.values():
    survey['boxcenter'] = np.array(survey['boxcenter'])
    survey['rotation'] = np.array(survey['rotation'])
