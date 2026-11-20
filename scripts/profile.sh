#!/usr/bin/env bash
# Nsight profiling of one control cycle: PyTorch eager bf16 vs direct TensorRT bf16.
#
#   scripts/profile.sh trace          nsys timelines (CUDA API, kernels, NVTX stages); no special rights
#   sudo scripts/profile.sh hw        hardware counters: ncu per-kernel metrics + nsys GPU-metrics sampling
#   sudo scripts/profile.sh ncu       only the ncu part of hw
#   scripts/profile.sh analyze        reports -> results/profile/profile.json (runs as the normal user)
#
# This box loads the driver with NVreg_RestrictProfilingToAdminUsers (RmProfilingAdminOnly: 1), so the
# GPU performance counters that ncu and `nsys --gpu-metrics-devices` read are root-only.
# Every run waits until no other process holds GPU memory (a profile taken next to another tenant
# measures the tenant, and ncu locks the GPU clocks for every process while it runs) and until the host has
# room for the run's memory cap, so it is safe to start on a busy box: it sits and waits.
# Big reports go to profiles/ (ignored by git); the extracted numbers go to results/profile/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD
mode=${1:?usage: scripts/profile.sh trace|hw|ncu|analyze}
NSYS=${NSYS:-/usr/local/bin/nsys}
NCU=${NCU:-/opt/nvidia/nsight-compute/2026.3.1/ncu}
ET15=$REPO/.venvs/et15/bin/python   # eager runs on the same env as results/summary.md's eager rows
TRT=$REPO/.venvs/trt/bin/python
mkdir -p profiles/nsys profiles/ncu logs results/profile

# One GPU job at a time, also between `trace` (user) and `hw` (root) started in different shells:
# every run holds profiles/.gpu.lock, then waits until no other process holds more than 1 GB of GPU memory
# and MemAvailable covers the run's MEM GB cap plus 2 GB (otherwise `limited` would shrink the cap to whatever
# is left, which on a full box is nothing).
LOCK=$REPO/profiles/.gpu.lock
[[ -e $LOCK ]] || { touch "$LOCK"; chmod 666 "$LOCK"; }
wait_for_box () {
  local busy avail need_kb=$(( (${MEM:-16} + 2) * 1024 * 1024 )) said_gpu=0 said_ram=0
  while :; do
    busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
           | grep -v -E 'gnome|Xorg|Xwayland' | awk -F', ' '$3 > 1024' || true)
    avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    [[ -z $busy ]] && (( avail >= need_kb )) && return 0
    if [[ -n $busy ]]; then
      (( said_gpu )) || { echo "[profile] $(date +%H:%M) waiting for the GPU (pid, name, MiB): $busy" >&2; said_gpu=1; }
    else
      (( said_ram )) || { echo "[profile] $(date +%H:%M) waiting for $(( need_kb >> 20 )) GB of free RAM (now $(( avail >> 20 )) GB)" >&2; said_ram=1; }
    fi
    sleep 30
  done
}

# Same guards as scripts/run_limited.sh (MemoryMax <= MemAvailable - 2 GB, no swap, OOM-first),
# but in the system slice when we are root (sudo has no systemd user manager).
limited () {
  local want_kb=$(( ${MEM:-16} * 1024 * 1024 ))
  local cap_kb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) - 2 * 1024 * 1024 ))
  (( cap_kb < want_kb )) && want_kb=$cap_kb
  local scope=(--user); [[ $EUID -eq 0 ]] && scope=()
  systemd-run "${scope[@]}" --scope -q -p MemoryMax="${want_kb}K" -p MemorySwapMax=0 -- choom -n 1000 -- "$@"
}

owner=${SUDO_USER:-$USER}
home=$(getent passwd "$owner" | cut -d: -f6)
export HF_HOME=${HF_HOME:-$home/.cache/huggingface} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO
export PATH=/usr/local/cuda/bin:$PATH
export DEBUGINFOD_URLS=   # nsys would otherwise fetch symbols for every loaded .so from debuginfod.ubuntu.com

run () {  # name, then command
  local name=$1 lockfd rc=0; shift
  exec {lockfd}<"$LOCK"; flock "$lockfd"
  wait_for_box
  echo "=== $name  ($(date +%H:%M:%S))"
  limited "$@" > "logs/$name.log" 2>&1 || rc=$?
  exec {lockfd}<&-
  echo "EXIT=$rc" >> "logs/$name.log"
  grep -E "^\[profile\]|==PROF==|Error|error:|Killed|EXIT" "logs/$name.log" | tail -6
  return $rc
}

EAGER=(-m pi05_edge.profile_run --path eager --dtype bfloat16)
TRTD=(-m pi05_edge.profile_run --path trt --dir "$REPO/models/trt")
NSYS_TRACE=(profile -f true --capture-range=cudaProfilerApi --capture-range-end=stop
            -t cuda,nvtx,osrt -s none --cpuctxsw=none --export=sqlite)

case $mode in
trace)
  run profile_nsys_eager "$NSYS" "${NSYS_TRACE[@]}" -o profiles/nsys/eager_bf16 "$ET15" "${EAGER[@]}"
  run profile_nsys_trt   "$NSYS" "${NSYS_TRACE[@]}" -o profiles/nsys/trt_bf16   "$TRT"  "${TRTD[@]}"
  ;;
hw|ncu)
  [[ $EUID -eq 0 ]] || { echo "[profile] $mode mode reads GPU performance counters: run it with sudo" >&2; exit 1; }
  trap 'chown -R "$owner:" profiles logs results/profile' EXIT   # hand root-written reports back, even on failure
  if [[ $mode == hw ]]; then
    # GPU-metrics sampling (SM / tensor / DRAM activity every 20 us) over 3 cycles, at the normal boost clocks.
    GM=(--gpu-metrics-devices=0 --gpu-metrics-set=gb20x --gpu-metrics-frequency=50000)
    run profile_gpumetrics_eager "$NSYS" "${NSYS_TRACE[@]}" "${GM[@]}" -o profiles/nsys/eager_bf16_gpumetrics "$ET15" "${EAGER[@]}" --cycles 3
    run profile_gpumetrics_trt   "$NSYS" "${NSYS_TRACE[@]}" "${GM[@]}" -o profiles/nsys/trt_bf16_gpumetrics   "$TRT"  "${TRTD[@]}" --cycles 3
  fi
  # ncu: every kernel of one cycle (vision, prefix, one denoising step; the other 9 steps replay the same
  # kernels), caches flushed between passes, clocks locked to base (ncu defaults).
  # Application replay: the whole program runs once per counter pass (31 passes) and kernels are matched by
  # name + grid. Kernel replay instead saves/restores every buffer a kernel can reach; with PyTorch's 17 GB
  # pool that grew ncu's own host memory to 12.5 GB and the cgroup OOM-killed it after 1,322 kernels.
  # The workload itself holds ~2 GB of host memory (weights stream straight to the GPU), hence a 12 GB cap.
  NCU_ARGS=(--replay-mode application --app-replay-match grid --app-replay-buffer file
            --profile-from-start off --target-processes all --nvtx --import-source no -f
            --section SpeedOfLight --section SpeedOfLight_RooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart
            --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats
            --section WarpStateStats --section SchedulerStats
            --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sector_hit_rate.pct,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_tf32_dst_fp32.sum)
  MEM=12 run profile_ncu_trt   "$NCU" "${NCU_ARGS[@]}" -o profiles/ncu/trt_bf16   "$TRT"  "${TRTD[@]}"  --warmup 2 --cycles 1 --steps 1 || echo "[profile] ncu trt failed; continuing" >&2
  MEM=12 run profile_ncu_eager "$NCU" "${NCU_ARGS[@]}" -o profiles/ncu/eager_bf16 "$ET15" "${EAGER[@]}" --warmup 2 --cycles 1 --steps 1 || echo "[profile] ncu eager failed" >&2
  ;;
analyze)
  "$ET15" -m pi05_edge.profile_analyze
  ;;
*)
  echo "unknown mode $mode" >&2; exit 2 ;;
esac
