"""Where a PPC campaign's files live.

Both entry points slice the same two conventions -- a trained experiment's path
and the per-draw output tree -- so they are spelled out once here instead of
being re-derived by positional indexing at each use. draw.py writes paths that
collect.py later reads, and the two silently disagreeing is the failure mode
this module exists to prevent.
"""

import os
from os.path import join

WDIR = '/work/hdd/bdne/maho3/cmass-ili'

# theta layout: 5 cosmology, then HOD (alphabetical, from hodprior.csv), then
# noise. draw.py injects by these slices and collect.py verifies against them.
N_COSMO = 5
N_NOISE = 2
COSMO_NAMES = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
NOISE_NAMES = ['noise_radial', 'noise_transverse']

# The stage-C job scripts run bias.hod.seed=1, and survey.aug_seed=1 for
# lightcones, so every draw's diagnostics land in hod00001[_aug00001].h5.
HOD_SEED = 1
AUG_SEED = 1


def parse_kcut(name):
    """Inverse of cmass.infer.tools.kcut_dirname.

    kmax is a scalar for a plain cut (kmin-0.0_kmax-0.4) or a per-summary
    mapping for a mixed one (kmin-0.0_kmax-Bk=0.2__Pk=0.4); resolve_kmax
    consumes either.
    """
    kmin, _, kmax = name.partition('_kmax-')
    kmin = kmin.replace('kmin-', '')
    if not kmax:
        raise ValueError(f'Cannot parse k-cut from {name!r}')
    if '=' not in kmax:
        return float(kmin), float(kmax)
    cuts = (part.split('=') for part in kmax.split('__'))
    return float(kmin), {('default' if k == 'def' else k): float(v)
                         for k, v in cuts}


def fmt_kmax(kmax):
    if not isinstance(kmax, dict):
        return str(kmax)
    return ', '.join(f'{k}<{v}' for k, v in sorted(kmax.items()))


def sim_subdir(cfg):
    """Per-draw simulation tree under a campaign dir, as the job scripts
    write it. Every stage runs with sim=fastpm; the box comes from the
    experiment being reproduced."""
    return join('fastpm', f'L{cfg.nbody.L}-N{cfg.nbody.N}')


class ExpPath:
    """A trained experiment.

    <wdir>/<suite>/<sim>/models/<tracer>/<summaries>/<kcut>. Stringifies and
    os.path.join()s as the path itself, so it substitutes for one anywhere a
    plain path was used.
    """

    def __init__(self, path):
        self.path = str(path).rstrip('/')
        parts = self.path.split(os.sep)
        if len(parts) < 6 or parts[-4] != 'models':
            raise ValueError(
                f'{self.path!r} is not <suite>/<sim>/models/<tracer>/'
                '<summaries>/<kcut>')
        self.suite, self.sim = parts[-6], parts[-5]
        self.tracer, self.summaries, self.kcut = parts[-3], parts[-2], parts[-1]
        self.kmin, self.kmax = parse_kcut(self.kcut)

    def __str__(self):
        return self.path

    def __fspath__(self):
        return self.path

    @property
    def is_lightcone(self):
        """A lightcone is already in redshift space, so its summaries carry no
        'z' prefix and live in their own diagnostics directory."""
        return self.tracer.endswith('_lightcone')

    @property
    def suite_root(self):
        """<wdir>/<suite>/<sim>, where this experiment's simulations live."""
        return self.path.split(os.sep + 'models' + os.sep)[0]

    @property
    def diag_dir(self):
        """Diagnostics directory within a simulation, as summ.py writes it."""
        return join('diag', self.tracer if self.is_lightcone else 'galaxies')

    def diag_file(self, hod_seed=HOD_SEED, aug_seed=AUG_SEED):
        """One draw's diagnostics, relative to its simulation directory."""
        aug = f'_aug{aug_seed:05d}' if self.is_lightcone else ''
        return join(self.diag_dir, f'hod{hod_seed:05d}{aug}.h5')

    def swap_suite(self, wdir, suite, sim):
        """The same tracer, summaries and k-cut under another suite, which is
        how infer.testing names an out-of-distribution experiment."""
        return ExpPath(join(wdir, suite, sim, 'models',
                            self.tracer, self.summaries, self.kcut))

    def campaign_dir(self, wdir, tag, testing=None):
        """Where a campaign's draws and outputs go. An out-of-distribution run
        gains a testing/<suite>_<sim>/ segment so it cannot collide with the
        self-consistent one."""
        root = join(wdir, 'ppc', f'{self.suite}_{self.sim}',
                    f'{self.summaries}_{self.kcut}')
        if testing is not None:
            root = join(root, 'testing', f'{testing.suite}_{testing.sim}')
        return join(root, tag)
