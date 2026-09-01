#!/bin/bash
#SBATCH --job-name=rd-h100-bench
#SBATCH --output=benchmarks/h100/slurm-%j.out
#SBATCH --error=benchmarks/h100/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# --- GPU request -----------------------------------------------------------
# Determined from `scontrol show partition` on rusty (read-only query):
#   partition `gpu`     : gres/gpu:a100-sxm4-80gb, gres/gpu:h100_pcie,
#                         gres/gpu:rtx_pro_6000_blackwell
#   partition `gpuxl`   : gres/gpu:h100 (SXM), gres/gpu:h200
#   partition `gpupreempt`: all of the above, preemptible
#
# The default below asks for ONE H100 PCIe in the regular `gpu` partition.
# NOTE (H100 PCIe vs SXM): PCIe is ~26 TFLOP/s FP64 with 2 TB/s HBM2e; the SXM
# part in `gpuxl` is ~34 TFLOP/s FP64 with 3.35 TB/s HBM3.  The predictions in
# PREDICTIONS.md are written for the PCIe part and are explicitly insensitive
# to this difference (the model is not FLOP- or bandwidth-bound), so either is
# a valid test -- but record which one you got (bench.py prints it).
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100_pcie:1
#
# ----> IF THIS FAILS with "Invalid generic resource specification", swap the
#       two lines above for ONE of:
#           #SBATCH --partition=gpuxl        # H100 SXM (may need an allocation)
#           #SBATCH --gres=gpu:h100:1
#       or
#           #SBATCH --partition=gpupreempt   # preemptible, always available
#           #SBATCH --gres=gpu:h100_pcie:1
#       Check what you are entitled to with:  sinfo -p gpu,gpuxl -o "%P %G %f"
#
# --- CPU / memory ----------------------------------------------------------
# 16 cores is deliberate, not arbitrary.  The A6000 study's CPU baseline was
# this workstation's 2x Xeon Gold 6244 = 16 PHYSICAL cores at 3.6 GHz.  Asking
# for 16 cores here makes the same-node CPU-vs-GPU comparison a like-for-like
# core-count fight rather than a handicap match, and it is also close to the
# per-GPU share of a 4-GPU rusty node (~88 cores / 4 = ~22), so it schedules
# easily.  Bump to 24-32 if you want to see whether the CPU side keeps scaling
# (the model's small triangular solves parallelise poorly, so it probably does
# not -- the OMP_NUM_THREADS=1 leg of the benchmark bounds that from below).
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -uo pipefail   # NOT -e: we want the trailing summary even on failure

VENV=/mnt/home/misi/src/ringdown/.venv
REPO=/mnt/home/misi/src/ringdown
cd "$REPO"

echo "=============================================================="
echo "host        : $(hostname)"
echo "date        : $(date)"
echo "job         : ${SLURM_JOB_ID:-<none>}"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?}"
echo "gpus        : ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-?}}"
echo "cwd         : $PWD"
echo "=============================================================="
nvidia-smi || { echo "FATAL: nvidia-smi failed -- no GPU visible"; exit 3; }
echo "--- driver / CUDA ---"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
echo "=============================================================="
# The pip CUDA wheels in the venv bundle their own CUDA 12.9 runtime, so only
# the NODE DRIVER matters.  CUDA 12.x minor-version compatibility needs driver
# >= 525.60.13; anything on rusty's H100 nodes is far newer.  Nothing needs to
# be module-loaded, and nothing must be installed.

# Let JAX/XLA and any BLAS see the cores Slurm actually gave us.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

OUT="$REPO/benchmarks/h100/results_${SLURM_JOB_ID:-manual}.json"

[ -x "$VENV/bin/python" ] || { echo "FATAL: no python at $VENV/bin/python"; exit 4; }

# One GPU, both precisions, plus the same-node CPU baseline.  bench.py fans out
# into its own subprocess legs (GPU f32, CPU f64 devtime, CPU f64 chains, CPU
# f64 OMP=1) because jax_enable_x64 and JAX_PLATFORMS are process-global.
#
# Run directly rather than under `srun`: this is a single task that forks its
# own children, and some Slurm configurations bind an srun step to one CPU
# unless -c is repeated, which would silently cripple the CPU baseline.  The
# sbatch script already runs inside the job's full allocation.
"$VENV/bin/python" -u "$REPO/benchmarks/h100/bench.py" \
    --platform gpu --out "$OUT" --tag "rusty-h100 job ${SLURM_JOB_ID:-manual}"
rc=$?

echo "=============================================================="
echo "bench.py exit code: $rc"
echo "results: $OUT"
echo "now run:  $VENV/bin/python $REPO/benchmarks/h100/analyze.py $OUT"
echo "(side files: <out>.gpu_f32.json / .cpu_f64.json / .cpu_f64_chains.json /"
echo " .cpu_f64_omp1.json / .f64ref.npz -- analyze.py needs only the main file,"
echo " which embeds them all)"
echo "=============================================================="
exit $rc
