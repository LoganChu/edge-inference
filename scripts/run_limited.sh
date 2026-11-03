#!/usr/bin/env bash
# Run a command inside a transient user cgroup with a hard memory ceiling.
#
# A runaway export (TensorRT builder, AOTInductor, a 14 GB fp32 state dict) is then
# OOM-killed on its own instead of swapping the whole desktop to death. Three guards:
#   1. MemoryMax on the cgroup, capped at (MemAvailable - 2 GB) at launch time, no swap;
#   2. oom_score_adj=1000 on the job so a *global* OOM would pick it, not another process;
#   3. HF_HUB_OFFLINE so nothing is ever downloaded mid-benchmark (the cache is warm).
#
#   MEM=16G scripts/run_limited.sh .venvs/trt/bin/python -m pi05_edge.export_trt ...
set -euo pipefail
MEM=${MEM:-16G}
want_kb=$(( ${MEM%G} * 1024 * 1024 ))
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
cap_kb=$(( avail_kb - 2 * 1024 * 1024 ))
if (( cap_kb < want_kb )); then
  echo "[run_limited] MemAvailable is $((avail_kb/1024/1024)) GB; capping MemoryMax at $((cap_kb/1024/1024)) GB instead of $MEM" >&2
  want_kb=$cap_kb
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}      # AOTInductor shells out to nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export PI05_EDGE_MEM_LIMIT="$((want_kb/1024/1024))G"
# --scope runs the command in this shell's environment; only the cgroup properties are added.
exec systemd-run --user --scope -q -p MemoryMax="${want_kb}K" -p MemorySwapMax=0 -- choom -n 1000 -- "$@"
