---
name: delta-slurm
description: Submit, monitor, and manage SLURM jobs on NCSA Delta (accounts, partitions, arrays, dependencies, holds, diagnostics, storage gotchas)
---

# SLURM on Delta (NCSA)

## Submission basics

- Account: `bdne-delta-cpu` (CPU), partitions: `cpu` (48 h max),
  `cpu-interactive` (1 h max, **short queue — good for stopgaps/tests**),
  `ghx4` (GPU). Job output convention:
  `#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out` (same for
  --error; %x=name, %A=jobid, %a=array index).
- Env setup in scripts: `source ~/.bashrc; conda activate cmass`. MPI codes
  may need `module load cray-mpich/8.1.32 gsl`.
- Useful submit flags: `--parsable` (returns bare jobid for scripting),
  `--hold` (submit held), `--export=ALL,VAR=x` (pass env vars),
  `--dependency=...`, `-t H:MM:SS` / `-p partition` (override script header).
- Multi-node MPI: launch with `srun -n N` inside the job (Cray systems; not
  mpirun). All files the ranks read/write must be on a SHARED filesystem —
  node-local /tmp is per-node and causes misleading fabric errors
  (`OFI poll failed ... Transport endpoint is not connected`) when ranks
  can't see input files. Genuine OFI errors also occur as transient
  fabric/node flakes — check whether failures repeat across node sets.

## Arrays

- `--array=100-199%10` = tasks 100..199, **max 10 running at once** (the
  throttle is per-array; two arrays with %10 each run 20 total).
- Change throttle on a pending array:
  `scontrol update job=<id> ArrayTaskThrottle=N`.
- FAILED/TIMEOUT/cancelled array tasks are consumed — SLURM won't rerun
  them. Resubmit as a new array with an explicit list:
  `sbatch --array=3155,3164,3190-3196%10 script.sh`.

## Dependencies — CRITICAL array gotcha

- `--dependency=after:<jobid>` = start after that job STARTS (not finishes);
  `afterok:` = after it succeeds; `afterany:` = after it ends.
- **Against an array job, `after:<arrayid>` expands to `after:<id>_*` = wait
  for ALL tasks to start** — with a throttle that means near the end, not
  the beginning. To gate on "the array began", depend on one early task:
  `--dependency=after:<id>_<first_index>`.
- Update in place while pending: `scontrol update job=<id> Dependency=...`
  (empty value clears). Circular dependencies are rejected — you cannot
  co-schedule two jobs mutually; gate the cheap/fast-scheduling job on the
  expensive one.

## Hold / release / cancel

- `scontrol hold <id>` — holds PENDING tasks only; running tasks continue.
  Shows as reason `JobHeldUser`. `scontrol release <id>` undoes.
- `scancel <id>` / `scancel <id>_<task>` for single array tasks.
- If a job is unexpectedly `JobHeldUser`, suspect an automated guard
  (watchdog script) — check before releasing.

## Monitoring & diagnostics

- Queue: `squeue -u maho3 -h -o "%i %j %T %M %r %R"` (state, elapsed,
  reason, nodes). squeue can time out when the controller is busy — treat
  errors as "unknown", not "no jobs" (check `$?` and empty output
  separately).
- History: `sacct -j <id> --format=JobID,Elapsed,State,NodeList -X`
  (`-X` = no substeps). Filter by name/time: `sacct --name=X -S <date>`.
- **Live memory of a running job**: `sstat -j <id>.batch --format=MaxRSS`.
  A job pinned at its `--mem` value with no progress = cgroup memory
  thrash. NOTE: the cgroup limit includes PAGE CACHE from the job's I/O,
  not just anon memory — heavy file copying counts against --mem.
- Node/job detail: `scontrol show job <id>` (Dependency, Reason, TRES).
- Log-watching beats polling: grep the job's .out file for progress
  markers; timestamps in logs identify stalls (e.g. "started reading, no
  output for 25 min").

## Storage (critical context for job design)

- `/work/hdd/bdne` — main shared Lustre; quota shared by the WHOLE
  allocation (`quota` command; `*` = over soft limit). **/scratch is the
  SAME filesystem** (same device, same quota), not a separate purged tier.
  Hitting the hard limit fails writes for every job in the allocation.
- `/tmp` on compute nodes — node-local NVMe (~740 GB CPU nodes), shared by
  jobs on that node, purged at job end. Fine for single-node work; never
  for multi-node inputs/outputs.
- `/projects/bdne` — small (500 G), for durable shared data, not job I/O.
- `du` on large Lustre trees is slow — use `timeout` around it; prefer
  counting files/dirs for quick state checks.
- Quota increases for /work: support request, "1-100 TB by allocation
  request" per docs.

## Patterns that work well

- **Watchdog loops** (run via background shell, NOT inside jobs): poll
  `quota`/`squeue` every 10-15 min; `scontrol hold` a producer array when
  storage nears a limit, release when consumers catch up. Remember these
  die with the Claude session — design jobs so an orphaned hold is safe,
  and journal state to memory for the next session.
- **Producer/consumer job pairs**: coordinate through sentinel files on the
  shared filesystem (`.done`/`.failed` touched by the job script after the
  payload command), never through job state alone — TIMEOUT kills leave no
  sentinel, so consumers must handle "producer vanished" (check sacct or
  apply timeouts) or they deadlock waiting.
- **Idempotent jobs** (skip-if-output-exists) make every recovery "just
  resubmit" — worth the small check at script start.
- Long campaigns: give service-style jobs generous idle-timeouts (producer
  queue stalls of hours are normal) and expect to resubmit them across
  walltime boundaries.
