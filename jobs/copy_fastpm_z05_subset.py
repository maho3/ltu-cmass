"""
Copy a z=0.5 (group "0.666660") subset of the abacuslike/fastpm/L2000-N256
suite to a new fastpm_z05/L2000-N256 suite.

Read-only against the source suite; only ever writes under DST_ROOT.
Each dest nbody.h5 gets: root-level attrs (if any) + the single group
"0.666660" (fvel, rho), copied natively via h5py so dtype/shape/attrs are
preserved bit-exact. config.yaml is copied unmodified alongside it.

Usage (one shard of a sharded run):
    python copy_fastpm_z05_subset.py --lhid-list lhid_list.txt \
        --shard-idx $SLURM_ARRAY_TASK_ID --num-shards 10 \
        --log copy_log_$SLURM_ARRAY_TASK_ID.tsv
"""
import argparse
import os
import shutil
import sys
import h5py

SRC_ROOT = "/ocean/projects/phy240015p/mho1/cmass-ili/abacuslike/fastpm/L2000-N256"
DST_ROOT = "/ocean/projects/phy240015p/mho1/cmass-ili/abacuslike/fastpm_z05/L2000-N256"
TARGET_GROUP = "0.666660"


def dest_is_valid(dst_h5_path):
    if not os.path.isfile(dst_h5_path):
        return False
    try:
        with h5py.File(dst_h5_path, "r") as f:
            if TARGET_GROUP not in f:
                return False
            g = f[TARGET_GROUP]
            return (
                "fvel" in g and "rho" in g
                and g["fvel"].shape == (256, 256, 256, 3)
                and g["rho"].shape == (256, 256, 256)
            )
    except Exception:
        return False


def copy_one(lhid):
    src_dir = os.path.join(SRC_ROOT, lhid)
    dst_dir = os.path.join(DST_ROOT, lhid)
    src_config = os.path.join(src_dir, "config.yaml")
    src_h5 = os.path.join(src_dir, "nbody.h5")
    dst_config = os.path.join(dst_dir, "config.yaml")
    dst_h5 = os.path.join(dst_dir, "nbody.h5")
    dst_h5_tmp = dst_h5 + ".tmp"

    if dest_is_valid(dst_h5) and os.path.isfile(dst_config):
        return "skip-already-done", ""

    os.makedirs(dst_dir, exist_ok=True)

    try:
        shutil.copy2(src_config, dst_config)

        if os.path.exists(dst_h5_tmp):
            os.remove(dst_h5_tmp)

        with h5py.File(src_h5, "r") as fsrc, h5py.File(dst_h5_tmp, "w") as fdst:
            for k, v in fsrc.attrs.items():
                fdst.attrs[k] = v
            fsrc.copy(fsrc[TARGET_GROUP], fdst, name=TARGET_GROUP)

        os.replace(dst_h5_tmp, dst_h5)
    except Exception as e:
        if os.path.exists(dst_h5_tmp):
            os.remove(dst_h5_tmp)
        return "error", str(e)

    if not dest_is_valid(dst_h5):
        return "error", "post-write validation failed"

    return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lhid-list", required=True)
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    with open(args.lhid_list) as f:
        all_lhids = [line.strip() for line in f if line.strip()]

    shard = all_lhids[args.shard_idx::args.num_shards]
    print(f"shard {args.shard_idx}/{args.num_shards}: {len(shard)} sims", flush=True)

    with open(args.log, "w") as logf:
        logf.write("lhid\tstatus\tdetail\n")
        for i, lhid in enumerate(shard):
            status, detail = copy_one(lhid)
            logf.write(f"{lhid}\t{status}\t{detail}\n")
            logf.flush()
            if status == "error":
                print(f"[{i+1}/{len(shard)}] {lhid}: ERROR - {detail}", flush=True)
            elif (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(shard)}] {lhid}: {status}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
