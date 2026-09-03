#!/bin/bash
#SBATCH --job-name=rd-gs-bench
# Slurm opens these BEFORE the script runs and cannot create the directory:
# `mkdir -p benchmarks/gs/results` from the repo root before sbatch (the
# in-script mkdir below is too late for these two files), and submit from the
# repo root (the paths are relative to the submission directory).
#SBATCH --output=benchmarks/gs/results/slurm-%j.out
#SBATCH --error=benchmarks/gs/results/slurm-%j.err
# Expected runtime ~4 h (local A6000 / Gold 6244 run: 5122 + 1285 + 970 + 1572
# + 1330 s = 2.9 h for the five original legs, of which the longdouble
# `reference` section alone was 3305 s; the cpu_f64_prod, gpu_f32_hi legs and
# the longdouble-Cholesky reference log-det add ~1 h; Icelake `gpu` nodes are
# not faster than the 3.6 GHz workstation for the CPU legs).  With REF set
# (see below) the reference section is skipped and ~1.5 h suffice.
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# One H100 PCIe in the regular `gpu` partition (see benchmarks/h100/submit.sbatch
# for the alternatives: --partition=gpuxl --gres=gpu:h100:1 for the SXM part, or
# --partition=gpupreempt).  Check entitlement with: sinfo -p gpu,gpuxl -o "%P %G %f"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100_pcie:1
#
# 16 cores: the multi-thread CPU legs use --cpu-omp = SLURM_CPUS_PER_TASK
# threads (BLAS and XLA pool), i.e. cpu_f64_omp16 / cpu_f32_omp16 here, next
# to cpu_f64_omp1 and the production-like cpu_f64_prod (1 BLAS thread, XLA
# pool = one thread per core).  Legs run one at a time; the GPU legs use one
# host thread (ringdown's import default OMP_NUM_THREADS=1), so no headroom
# is needed for them.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Gohberg-Semencul vs Cholesky benchmark (benchmarks/gs) on an H100.
# Submit FROM THE REPO ROOT after running prep_inputs.py once:
#     cd /path/to/ringdown
#     PYTHONPATH=$PWD .venv/bin/python benchmarks/gs/prep_inputs.py --out benchmarks/gs/inputs
#     mkdir -p benchmarks/gs/results
#     sbatch benchmarks/gs/submit_h100.sbatch
# The inputs (~6.4 GB of npz) are NOT rebuilt here: prep needs the GW150914
# data files under data/ and takes ~80 min on 8 cores with the default NUTS
# warmups (~5 min with --no-warmup); do it on a workstation.  Inputs written
# by an older prep_inputs.py must be upgraded first (minutes):
#     PYTHONPATH=$PWD .venv/bin/python benchmarks/gs/prep_inputs.py --refresh-precompute --out benchmarks/gs/inputs
#
# Optional: REF=/path/to/local.ref.npz reuses an existing extended-precision
# reference (numpy longdouble is platform-independent on x86-64) and drops the
# `reference` section, saving ~1 h; the reference must have been produced from
# the same inputs (the kit checks the fixed points and ignores a stale one).
#
# The venv's CUDA comes from the pip wheels (jax-cuda12-plugin, nvidia-*-cu12);
# do not `module load cuda` (see benchmarks/h100/README.md).  gpu_f32 runs with
# JAX's default matmul precision (TF32 dots, as production does); gpu_f32_hi
# repeats the f32 accuracy section with --matmul-precision highest, which is
# the leg the float32 route-accuracy comparison is read from.

set -uo pipefail   # not -e: keep the trailing summary on failure

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$REPO/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
INPUTS="${INPUTS:-$REPO/benchmarks/gs/inputs}"
REF="${REF:-}"
cd "$REPO" || { echo "FATAL: cannot cd to REPO=$REPO"; exit 2; }
[ -f "$REPO/benchmarks/gs/bench_gs.py" ] || {
    echo "FATAL: REPO=$REPO has no benchmarks/gs/bench_gs.py; submit from the repo root"
    exit 2
}
[ -x "$PYTHON" ] || { echo "FATAL: no interpreter at PYTHON=$PYTHON (set PYTHON= or VENV=)"; exit 4; }
[ -f "$INPUTS/aligo02_d2_n205_m2.npz" ] || {
    echo "FATAL: no prepared inputs in $INPUTS; run prep_inputs.py first (see header)"
    exit 5
}
if [ -n "$REF" ] && [ ! -f "$REF" ]; then
    echo "FATAL: REF=$REF does not exist"
    exit 6
fi

echo "=============================================================="
echo "host  : $(hostname)     date: $(date)"
echo "job   : ${SLURM_JOB_ID:-<none>}  cpus: ${SLURM_CPUS_PER_TASK:-?}  mem: ${SLURM_MEM_PER_NODE:-?}"
echo "gpus  : ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-?}}"
echo "=============================================================="
nvidia-smi || { echo "FATAL: nvidia-smi failed -- no GPU visible"; exit 3; }
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv

# measure the ringdown in $REPO, not whatever the venv has installed
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# each leg sets its own OMP_NUM_THREADS / NPROC (bench_gs.py --omp / --xla-threads)
# before importing jax; nothing is exported here

mkdir -p "$REPO/benchmarks/gs/results"
OUT="$REPO/benchmarks/gs/results/results_${SLURM_JOB_ID:-manual}.json"
BASE="${OUT%.json}"
CPU_OMP="${SLURM_CPUS_PER_TASK:-8}"

SECTIONS="env,reference,correctness,devtime,compile,f32acc,nuts"
REF_ARGS=()
if [ -n "$REF" ]; then
    SECTIONS="env,correctness,devtime,compile,f32acc,nuts"
    REF_ARGS=(--ref "$REF")
    echo "reusing reference $REF; sections: $SECTIONS"
fi

# Parent process: orchestrates cpu_f64_omp${CPU_OMP}, cpu_f64_omp1, cpu_f64_prod,
# cpu_f32_omp${CPU_OMP}, gpu_f64, gpu_f32, gpu_f32_hi as child processes
# (precision and platform are process-global in JAX).  Run directly, not under
# srun (single task that forks its own children; some srun configurations bind
# the step to one CPU).  `nuts` must be listed explicitly; it runs in the
# cpu_f64_omp${CPU_OMP} leg only.  --leg-timeout bounds ONE leg: the reference-
# producing leg took 5122 s locally with 8 threads (a leg that times out is
# discarded entirely: its JSON is written only at the end).
"$PYTHON" -u "$REPO/benchmarks/gs/bench_gs.py" \
    --platform gpu --inputs "$INPUTS" --out "$OUT" --nfft both \
    --sections "$SECTIONS" --cpu-omp "$CPU_OMP" "${REF_ARGS[@]}" \
    --leg-timeout 10800 --tag "rusty-h100 job ${SLURM_JOB_ID:-manual}"
rc=$?

echo "=============================================================="
echo "bench_gs.py exit code: $rc"
echo "results: $OUT"
echo "  side files: ${BASE}.<leg>.json (one per leg), ${BASE}.ref.npz (reference dump)"
echo "now run:  $PYTHON $REPO/benchmarks/gs/analyze_gs.py $OUT --md > ${BASE}.md"
echo "=============================================================="
exit $rc
