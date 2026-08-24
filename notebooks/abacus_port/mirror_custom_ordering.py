#!/usr/bin/env python
"""Create abacus/nbodyconc/L2000-N256/{lhid}/ as a full COPY of abacus/nbodyall,
using the SAME lhid ordering AND coverage as abacus/custom (both derive from
/anvil/scratch/x-mho1/cmass-ili/scratch/abacus_custom_table.csv, the user's own
manually-ported table, lhid 0-118) -- i.e. nbodyconc is a drop-in, same-lhid
companion to custom that adds the NFW-fit concentration field, not a different
indexing scheme.

nbodyall's own table (abacus_custom_table.csv, alongside this script) has since been
extended with 19 more rows (lhid 119-137, c000 ph006-ph024) once that phase data was
transferred separately -- nbodyall covers those too, but nbodyconc deliberately does
NOT, since custom was never regenerated for them. This script mirrors only the lhids
actually present in abacus/custom (read from disk, not from the table), so reruns
stay capped to custom's real coverage even if the table grows again later.

Copies (not symlinks) so nbodyconc stands on its own; nbodyall's files are untouched
(read-only source).
"""
import os
import shutil
from glob import glob
from os.path import join, exists, islink, basename
from port_abacus_lib import OUT_TMPL

CUSTOM_DIR = '/anvil/scratch/x-mho1/cmass-ili/abacus/custom/L2000-N256'
NBODYCONC_TMPL = '/anvil/scratch/x-mho1/cmass-ili/abacus/nbodyconc/L2000-N256/{lhid}'

custom_lhids = sorted(int(basename(p)) for p in glob(f'{CUSTOM_DIR}/*') if basename(p).isdigit())

made = skipped_no_data = 0
for lhid in custom_lhids:
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

print(f'nbodyconc: {made} lhid dirs copied from nbodyall (capped to custom\'s '
      f'{len(custom_lhids)} lhids), skipped {skipped_no_data} not-yet-ported in nbodyall')
