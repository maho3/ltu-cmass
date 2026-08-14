#!/usr/bin/env python
"""Create abacus/nbodyconc/L2000-N256/{lhid}/ as a full COPY of abacus/nbodyall,
using the SAME lhid ordering as abacus/custom (both derive from abacus_custom_table.csv,
0-118 continuous, extended to 0-137 with the appended c000 ph006-ph024 rows) -- i.e.
nbodyconc is a drop-in, same-lhid companion to custom that adds the NFW-fit
concentration field, not a different indexing scheme. Copies (not symlinks) so
nbodyconc stands on its own; nbodyall's files are untouched (read-only source).
"""
import os
import shutil
from os.path import join, exists, islink
from port_abacus_lib import SIMNAME, OUT_TMPL

NBODYCONC_TMPL = '/anvil/scratch/x-mho1/cmass-ili/abacus/nbodyconc/L2000-N256/{lhid}'

made = skipped_no_data = 0
for lhid in sorted(SIMNAME):
    src = OUT_TMPL.format(lhid=lhid)
    if not exists(join(src, 'halos.h5')):
        skipped_no_data += 1
        continue
    dst = NBODYCONC_TMPL.format(lhid=lhid)
    os.makedirs(dst, exist_ok=True)
    for fn in ('halos.h5', 'config.yaml'):
        link = join(dst, fn)
        if islink(link):
            os.remove(link)      # replace a stale symlink with a real copy
        shutil.copy2(join(src, fn), link)
    made += 1

print(f'nbodyconc: {made} lhid dirs copied from nbodyall (same lhid as custom), '
      f'skipped {skipped_no_data} not-yet-ported')
