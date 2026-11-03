#!/usr/bin/env bash
# The full matrix as it was finally run: strictly one GPU job at a time, every step inside the memory cgroup.
# Logs -> logs/, machine-readable results -> results/raw/, tables -> results/summary.md.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ET15=.venvs/et15/bin/python
TRT=.venvs/trt/bin/python
run () {  # name, then command
  local name=$1; shift
  echo "=== $name  ($(date +%H:%M:%S))"
  MEM=${MEM:-16G} scripts/run_limited.sh "$@" > "logs/$name.log" 2>&1
  local rc=$?
  echo "EXIT=$rc" >> "logs/$name.log"
  grep -E "^\[(time|fidelity|save|plan|pte|trt|lower|result)\]|Error|error:|Killed" "logs/$name.log" | tail -12
  return $rc
}
mkdir -p logs
$ET15 -m pi05_edge.data --n-obs 8

# --- references
run eager_bf16          $ET15 -m pi05_edge.bench_eager --dtype bfloat16
run eager_fp32          $ET15 -m pi05_edge.bench_eager --dtype float32
run eager_bf16_tf32     $ET15 -m pi05_edge.bench_eager --dtype bfloat16 --tf32

# --- path 1: ExecuTorch + CUDA (AOTInductor), ExecuTorch 1.5.1 (device-resident inputs; outputs via host, see README)
for m in vision prefix denoise_step denoise10; do
  run export_et_cuda_15_$m $ET15 -m pi05_edge.export_et_cuda --method $m --out models/et_cuda_1.5.1 --io device-in
done
run bench_et_cuda_15    $ET15 -m pi05_edge.bench_et --dir models/et_cuda_1.5.1 --label et_cuda_1.5.1

# --- paths 2+3: TensorRT engines (direct .plan and ExecuTorch .pte). The 2.5 B prefix is exported in 4 chunks
#     because the TensorRT builder and the .pte embedding need host RAM proportional to the engine.
for m in vision prefix_embed prefix_l0 prefix_l1 prefix_l2 denoise_step denoise10; do
  run export_trt_$m       $TRT -m pi05_edge.export_trt --method $m --out models/trt
done
run package_pte_vision  $TRT -m pi05_edge.package_pte --dir models/trt/vision      # .pte in a fresh process (memory)
run bench_trt_direct    $TRT -m pi05_edge.bench_trt_direct --dir models/trt --label trt_direct
run bench_et_trt        $TRT -m pi05_edge.bench_et --dir models/trt --label et_trt_1.4.1

# --- path 1 again on the ExecuTorch 1.4.1 runtime that Torch-TensorRT bundles (same runtime as path 3, host I/O)
for m in vision prefix denoise_step denoise10; do
  run export_et_cuda_14_$m $TRT -m pi05_edge.export_et_cuda --method $m --out models/et_cuda_1.4.1 --io host --split-to-slices
done
run bench_et_cuda_14    $TRT -m pi05_edge.bench_et --dir models/et_cuda_1.4.1 --label et_cuda_1.4.1
mkdir -p models/et_cuda_1.4.1_chunked
for m in vision denoise_step denoise10; do ln -sfn "$PWD/models/et_cuda_1.4.1/$m" "models/et_cuda_1.4.1_chunked/$m"; done
for m in prefix_embed prefix_l0 prefix_l1 prefix_l2; do
  run export_et_cuda_14_$m $TRT -m pi05_edge.export_et_cuda --method $m --out models/et_cuda_1.4.1_chunked --io host --split-to-slices
done
run bench_et_cuda_14_chunked $TRT -m pi05_edge.bench_et --dir models/et_cuda_1.4.1_chunked --label et_cuda_1.4.1_chunked

# --- precision-controlled vision variants
run export_trt_vision_fp32        $TRT  -m pi05_edge.export_trt --method vision --out models/trt_fp32 --disable-tf32 --skip-pte
run bench_trt_vision_fp32         $TRT  -m pi05_edge.bench_trt_direct --dir models/trt_fp32 --label trt_direct_vision_fp32 --methods vision --vision-only
run export_et_cuda_15_vision_tf32 $ET15 -m pi05_edge.export_et_cuda --method vision --out models/et_cuda_1.5.1_tf32 --tf32
run bench_et_cuda_15_vision_tf32  $ET15 -m pi05_edge.bench_et --dir models/et_cuda_1.5.1_tf32 --label et_cuda_1.5.1_vision_tf32 --methods vision --vision-only

$ET15 -m pi05_edge.report
