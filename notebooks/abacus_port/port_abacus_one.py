#!/usr/bin/env python
"""Port one AbacusSummit lhid to ltu-cmass format with NFW-fit c200c.
Usage:  python port_abacus_one.py <lhid> [--overwrite]
Intended as the body of a SLURM array (one sim per task). Resumable.
"""
import sys
import time
from port_abacus_lib import get_index, process_lhid, SIMNAME

if __name__ == '__main__':
    lhid = int(sys.argv[1])
    overwrite = '--overwrite' in sys.argv[2:]
    if lhid not in SIMNAME:
        print(f'lhid {lhid} not in table; nothing to do'); sys.exit(0)
    t0 = time.time()
    tf_dict, index = get_index()                   # loads cached pickle (fast) or builds once
    print(f'lhid {lhid} ({SIMNAME[lhid]}): index ready ({time.time()-t0:.0f}s), processing...',
          flush=True)
    status, _, n = process_lhid(lhid, index, tf_dict, overwrite=overwrite, verbose=True)
    print(f'lhid {lhid}: {status} {n if n is not None else ""} halos  ({time.time()-t0:.0f}s)')
