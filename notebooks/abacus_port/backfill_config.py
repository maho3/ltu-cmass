#!/usr/bin/env python
"""Backfill config.yaml into every already-ported abacus/nbodyall lhid directory
(this only writes the small yaml; it does not touch halos.h5 or re-run the port)."""
from os.path import join, exists
from port_abacus_lib import OUT_TMPL, SIMNAME, write_config

done = skip = 0
for lhid in sorted(SIMNAME):
    outdir = OUT_TMPL.format(lhid=lhid)
    if not exists(join(outdir, 'halos.h5')):
        continue
    write_config(outdir, lhid)
    done += 1

print(f'wrote config.yaml to {done} lhid dirs')
