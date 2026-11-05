# edge-inference — π₀.5 through three edge deployment stacks

One trained VLA, **π₀.5** (`lerobot/pi05_libero_base`, the PyTorch port of Physical Intelligence's
`pi05_libero` checkpoint: 3.6 B parameters, SigLIP So400m + Gemma-2B prefix + Gemma-300M flow-matching
action expert), exported through three deployment paths and measured on the same machine with the
same observations:

| path | compiler (export time) | runtime (robot side) | artifact |
|---|---|---|---|
| **ExecuTorch + CUDA** | `torch.export` → ExecuTorch `CudaPartitioner` → **AOTInductor** (Triton kernels, cuBLAS shims) | ExecuTorch runtime + `libaoti_cuda_backend` | `.pte` (program + AOTI `.so`) and `.ptd` (weights) |
| **Direct TensorRT** | `torch.export` → Torch-TensorRT dynamo converters → **TensorRT builder** | plain TensorRT runtime (`execute_async_v3`), no PyTorch, no ExecuTorch | `.plan` engine |
| **ExecuTorch + TensorRT** | the *same* TensorRT engine, wrapped as an ExecuTorch delegate | ExecuTorch runtime + TensorRT delegate | `.pte` with the engine embedded |

PyTorch eager (fp32 and bf16) is the reference for both numerics and latency.

Results and the discussion are in [`results/summary.md`](results/summary.md) and the **Findings**
section below. Every number in this repo was produced by the scripts here; nothing is quoted from
vendor material.

## How the model is cut for export

A robot controller calls π₀.5 as *encode once, denoise ten times*. The policy is exported as four
static-shape graphs (`pi05_edge/model.py`), which is also how each path is timed:

```
vision        (3,3,224,224) f32  ->  image tokens (3,256,2048)          SigLIP + projector, fp32 (as trained)
prefix        image tokens, 200 text tokens, pad mask  ->  K,V (18,1,1,968,256) bf16   Gemma-2B prefill
denoise_step  K,V, pad mask, x_t (1,50,32), t  ->  v_t (1,50,32)        Gemma-300M expert, one Euler step
denoise10     K,V, pad mask, noise  ->  actions (1,50,32)               all 10 Euler steps fused in one graph
```

Shapes are fixed: three cameras (two real LIBERO views plus the one *empty camera* this checkpoint was
trained with), the tokenizer's 200-token prompt budget, so the prefix is 968 positions. The two
denoising variants bracket the deployment design space: `denoise_step` crosses the runtime boundary
ten times per control cycle (18 MB of KV cache in, 6 KB out, each time); `denoise10` crosses it once.

The inputs are eight real LIBERO scenes (agent + wrist camera, proprio state, instruction) rendered for
an earlier benchmark, tokenised with the PaliGemma sentencepiece model and normalised with openpi's
`pi05_libero` statistics, so the prompts are exactly what the policy saw in training
(`pi05_edge/data.py`). The same noise sample is used everywhere, so action chunks are directly
comparable across paths.

## Findings

Numbers are medians over 20 control cycles on an RTX 5090 (CUDA 13.2) while another process was
using 13 of the 16 cores; full tables with p90, fidelity, export cost and memory are in
[`results/summary.md`](results/summary.md).

### Headline: one control cycle (encode + 10 denoising steps), batch 1

| path | vision | prefix | 1 expert step | control cycle | vs eager bf16 |
|---|---|---|---|---|---|
| PyTorch eager fp32 (LeRobot `sample_actions`) | 29.6 | 89.3 | 11.6 | 246 ms | — |
| PyTorch eager bf16 | 29.1 | 29.9 | 13.4 | 189 ms | 1.0× |
| **ExecuTorch + CUDA** (AOTInductor, ET 1.5.1, Python runtime) | 29.3 | 25.9 | **2.8** | **92 ms** | 2.1× |
| **Direct TensorRT** (11.1, raw runtime) | **9.3** | 26.2 | **3.0** | **62 ms** | 3.0× |
| **ExecuTorch + TensorRT** (delegate, ET 1.4.1, Python runtime) | 11.6 | 39.2 | 5.4 | 110 ms | 1.7× |
| ExecuTorch + CUDA on the same ET 1.4.1 runtime, host I/O | 33.5 | 28.3 | 5.5 | 124 ms (99 ms with the fused 10-step unit) | 1.5× / 1.9× |
| ExecuTorch + CUDA, ET 1.4.1, prefix split in the same 4 chunks as TensorRT | 32.6 | 38.2 | 5.5 | 129 ms (104 ms fused) | 1.5× / 1.8× |

All compiled paths reproduce the bf16 reference to bf16 noise (action MAE ≈ 0.0009 vs eager bf16,
≈ 0.002 vs eager fp32, identical gripper signs); the fp32 eager run reproduces LeRobot's own loop
exactly (MAE 0).

### What the numbers say

**1. The win is in the action expert, and all compilers get it.** The Gemma-300M expert step is
13 ms in eager PyTorch and about 3 ms under both AOTInductor and TensorRT: ten steps go from 134 ms
to 30 ms. Eager PyTorch is launch-bound here (18 layers × dozens of small kernels on a 50-token
sequence), and fusion plus static shapes remove that. The 2.5 B-parameter prefix prefill is
GEMM-bound and already runs at tensor-core speed in eager bf16, so every path lands at 27–30 ms.
Fusing the ten steps into one graph (`denoise10`) buys nothing more once the step itself is
compiled: 26.5 ms for ten fused steps vs 27.6 ms for ten calls.

**2. "fp32" means different things.** TensorRT computes fp32 layers in TF32 by default; PyTorch
and AOTInductor do not. That alone is the 29 → 9.3 ms vision gap: eager PyTorch with TF32 enabled
runs the same SigLIP tower in 16 ms, and the action error is unchanged (MAE 0.0021 either way).
The remaining 16 → 9.3 ms is TensorRT's attention/GEMM fusion. ExecuTorch's CUDA backend is no
faster than eager on the fp32 vision tower (29 to 34 ms across runs) because its Triton attention
kernel is bf16-only and the fp32 tower falls back to a decomposed attention. The TF32-controlled variants
isolate the two effects on the vision tower alone:

| SigLIP + projector, 3 images | true fp32 | TF32 |
|---|---|---|
| PyTorch eager | 29.1 ms | 16.0 ms |
| ExecuTorch + CUDA (AOTInductor) | 29 to 34 ms | 13.0 ms |
| Direct TensorRT | 14.3 ms | 9.3 ms |

At equal precision TensorRT is 2× faster than eager on this tower and AOTInductor is on par with or
slightly ahead of eager; the headline 3× vision gap in the main table is two-thirds precision.

**3. The runtime boundary decides whether a delegate's speed reaches the robot.** The same
TensorRT engines that run a cycle in 62 ms from the raw runtime take 110 ms through the
ExecuTorch 1.4.1 Python runtime that Torch-TensorRT 2.14 ships, because that runtime copies every
input to the host and every output back (expert step 3.0 → 5.4 ms: the 18 MB KV cache crosses
PCIe twice per step; prefix 26 → 39 ms). Running ExecuTorch+CUDA through the *same* 1.4.1 runtime
with the same host copies shows that the boundary is the whole cost: the expert step becomes
5.5 ms and the cycle 124 ms there, splitting the prefix into the four chunks TensorRT needed costs
another 10 ms (28 → 38 ms), and with that the TensorRT delegate's own per-call handling is not
measurable (prefix 38 vs 39 ms, step 5.5 vs 5.4 ms). What separates the two delegates on this
runtime is TensorRT's vision tower (12 vs 33 ms). On a host-copy runtime the fused 10-step unit
finally pays off, because the KV cache is uploaded once instead of ten times (124 → 99 ms).
ExecuTorch 1.5.1's runtime can take device-resident *inputs* but its Python bindings still
allocate outputs on the host; fully device-resident I/O is a C++-only fast path (the `Module`
API, shipped as a C++ SDK in the 1.5.1 wheel). The ET+CUDA figure of 92 ms therefore still
carries ~10 ms of host round trips a C++ deployment would not pay. Direct TensorRT has no such
boundary: engines are bound to device pointers and launched on one stream. The host-copy paths
are also the ones that move with the other tenant's CPU load: the same ET+TensorRT engines
measured 166 ms in an earlier session and 110 ms in the session all tables here come from.

**4. Memory is where the three stacks differ most, and it is export-time memory.**

| | ExecuTorch + CUDA | Direct TensorRT | ExecuTorch + TensorRT |
|---|---|---|---|
| whole 2.5 B prefix as one unit | yes: 4.8 GB `.ptd`, 4 MB `.pte`, 29 s compile | builder needs weights + engine in host RAM (≈ 2× bf16 size) → split into 4 engines on this box | `.pte` embeds the engine through a base64 round trip: ≈ 3× engine size *on top of* the builder's leftovers; 1.66 GB vision engine was OOM-killed at 11 GB |
| 10-step fused expert | 1.1 GB, same weight file as one step, 243 s compile | OOM-killed at 11 GB host (22.6k-node network) | n/a |
| artifact size | weights once, programs 4–9 MB | one `.plan` per engine, bf16 kept (1.3 GB per 6 layers) | `.pte` = engine size |
| workaround used here | — | chunked prefix (embedding + 3×6 layers) | chunked prefix + re-package `.plan` → `.pte` in a fresh process (`package_pte.py`) |

AOTInductor streams weights straight to the `.ptd`, so the only export-time constraint was GPU
memory. TensorRT's builder and, worse, the ExecuTorch embedding of the engine, need host RAM
proportional to the engine, which is what forced the chunked prefix and the separate packaging
step on a 30 GB machine that another process was using half of.

**5. Graph-level friction is small but real, and TensorRT-specific.** `torch.export` captured all
four units of the LeRobot π₀.5 unchanged (non-strict mode; 1.4k / 2.5k / 3.1k / 24.8k nodes).
AOTInductor lowered them as-is. Torch-TensorRT refused 72 `aten.cat` nodes per expert step: the
`cat([torch.tensor([]), K])` that transformers' `DynamicCache` emits when it lazily initialises a
layer, legal in PyTorch, rank-mismatched for TensorRT. A six-line graph pass folds them away
(`pi05_edge/graph_passes.py`, bit-identical output). TensorRT also reorders engine input bindings
relative to the forward signature, so a raw-runtime caller must bind by name.

### FP8

Same three stacks, with the 252 Gemma linear layers (q/k/v/o and gate/up/down in the 2B prefix and
the 300M expert) in FP8 E4M3 W8A8. Vision, the token embedding, the (Ada)RMSNorm dense layers and
the action/time projections keep their precision. There is no FP8 format shared across the stacks,
so each uses its own quantiser (`pi05_edge/quantize.py`):

| stack | FP8 recipe | status |
|---|---|---|
| PyTorch eager | torchao `Float8DynamicActivationFloat8Weight`, per-row dynamic scales, `torch._scaled_mm` | runs |
| Direct TensorRT / ExecuTorch + TensorRT | NVIDIA ModelOpt `FP8_DEFAULT_CFG`: per-tensor static scales, max-calibrated on 40 LIBERO scenes disjoint from the 8 evaluation scenes, exported as `tensorrt.quantize_op` Q/DQ nodes | runs; same engines in both packagings |
| ExecuTorch + CUDA | torchao float8 (W8A8 and weight-only) | **not supported in ET 1.5.1**: `_scaled_mm` needs a column-major weight view that ExecuTorch's copy-only edge dialect cannot express ("mat2 must be col_major"), and float8 weights fail the CUDA backend's weight-metadata check at load (`Malformed CUDA FQN weight metadata`, error 0x23). The backend's low-precision path is INT4 weight-only (torchao `tile_packed_to_4d`), which does lower and run. |

| path | prefix | 1 expert step | 10 fused steps | control cycle | action MAE vs eager fp32 |
|---|---|---|---|---|---|
| PyTorch eager bf16 (reference) | 29.9 | 13.4 | 115 | 189 ms | 0.0021 |
| PyTorch eager FP8 (torchao) | 25.0 | 24.4 | 248 | 307 ms | 0.0042 |
| Direct TensorRT bf16 | 26.2 | 3.0 | not built | 62 ms | 0.0019 |
| **Direct TensorRT FP8** | **15.9** | 2.9 | 26.2 | **53 ms** (50 ms fused) | 0.0059 |
| ExecuTorch + TensorRT bf16 (ET 1.4.1, host I/O) | 39.2 | 5.4 | not built | 110 ms | 0.0019 |
| ExecuTorch + TensorRT FP8 (ET 1.4.1, host I/O) | 25.9 | 5.7 | 31.1 | 96 ms (70 ms fused) | 0.0059 |

What FP8 changes, and does not:

* **It helps exactly one stage.** The GEMM-bound prefix drops 26 → 16 ms in TensorRT. The expert step
  is launch-bound and does not gain: 3.0 → 2.9 ms. TensorRT's layer profiler shows why: at 50 tokens
  the step is 54 % fused-attention kernels, which stay bf16, and FP8 Q/DQ breaks the horizontal fusion
  of q/k/v and gate/up, so the step runs 131 GEMM kernels instead of 77 and 484 layers instead of 210.
  The whole cycle goes 62 → 53 ms. FP8 is a prefill optimisation for this model.
* **In eager PyTorch FP8 is a loss.** torchao's dynamic quantisation adds amax, scale and cast kernels
  around every linear; on the launch-bound expert that doubles the step (13 → 24 ms) and the cycle
  ends at 307 ms, slower than bf16. FP8 needs a compiler that fuses the quantisation into the GEMM.
* **It halves what the memory-bound steps had to move.** Engines: expert 1.10 GB → 0.79 GB, each
  6-layer prefix chunk 1.32 → 0.66 GB. The smaller engines also pass through the ExecuTorch `.pte`
  serialisation in-process (2 s each) where the bf16 ones needed the separate packaging step, and the
  fused ten-step engine that was OOM-killed in bf16 now builds (294 s). It comes out at 358 MB:
  with the denoising schedule fixed inside the graph, the time-conditioning path (sinusoidal
  embedding, time MLP and the 453 MB of fp32 AdaRMS modulation weights) is a compile-time constant,
  and TensorRT folds it away. That is the strongest argument for the fused unit on this model.
* **Accuracy cost is real but small.** Action MAE against the fp32 reference goes from 0.0021 (bf16,
  any path) to 0.0059 (TensorRT FP8, static scales) and 0.0042 (torchao, dynamic per-row scales);
  max error 0.06 on actions of mean magnitude 0.32, gripper signs unchanged on all 8 scenes. Modular
  and fused FP8 agree to 1e-4, so compounding ten FP8 steps does not add error. The eager ModelOpt fake-quant pass (same scales, simulated rounding, no FP8 kernels) lands at 0.0058 vs fp32, and the TensorRT engine sits 0.0039 MAE from it, the same distance two eager FP8 runs sit from each other (0.0038 to 0.0044), so the error is the static per-tensor recipe itself, not the engine.
* **Export-time friction is unchanged.** The Q/DQ graph needs the same empty-`cat` pass (72 nodes per
  step, 720 in the fused graph) and the same name-based binding; calibration adds about 20 s per export.

### FP4 (NVFP4)

Same 252 linears again, this time in NVFP4: E2M1 values in blocks of 16, an E4M3 scale per block
and one fp32 scale per tensor, for weights and activations (W4A4). TensorRT runs FP4 GEMMs only on
Blackwell (this RTX 5090 is sm_120). Recipes, again one per stack (`pi05_edge/quantize.py`):

| stack | FP4 recipe | status |
|---|---|---|
| PyTorch eager | torchao `NVFP4DynamicActivationNVFP4Weight`, dynamic block scales, `torch._scaled_mm` FP4; `use_triton_kernel=False` because the MSLK Triton quantiser it wants is not installed, so the block quantisation runs as plain PyTorch ops | runs |
| Direct TensorRT / ExecuTorch + TensorRT | NVIDIA ModelOpt `NVFP4_DEFAULT_CFG`, per-tensor scales max-calibrated on the same 40 LIBERO scenes, exported as `tensorrt.dynamic_block_quantize_op` nodes | runs; same engines in both packagings |
| ExecuTorch + CUDA | — | **not supported**: the 1.5.1 CUDA backend has no FP4 or FP8 kernels at all; its low-bit paths are integer weight-only (INT4/INT5/INT6/INT8 Triton dispatch) |

| path | prefix | 1 expert step | 10 fused steps | control cycle | action MAE vs eager fp32 |
|---|---|---|---|---|---|
| PyTorch eager bf16 (reference) | 29.9 | 13.4 | 115 | 189 ms | 0.0021 |
| PyTorch eager FP8 (torchao) | 25.0 | 24.4 | 248 | 307 ms | 0.0042 |
| PyTorch eager FP4 (torchao, no Triton quantiser) | 50.9 | 46.0 | 455 | 540 ms | 0.0209 |
| Direct TensorRT bf16 | 26.2 | 3.0 | not built | 62 ms | 0.0019 |
| Direct TensorRT FP8 | 15.9 | 2.9 | 26.2 | 53 ms (50 ms fused) | 0.0059 |
| **Direct TensorRT FP4** | **8.8** | 3.4 | 30.9 | **51 ms** (47 ms fused) | 0.0196 |
| ExecuTorch + TensorRT bf16 (ET 1.4.1, host I/O) | 39.2 | 5.4 | not built | 110 ms | 0.0019 |
| ExecuTorch + TensorRT FP8 (ET 1.4.1, host I/O) | 25.9 | 5.7 | 31.1 | 96 ms (70 ms fused) | 0.0059 |
| ExecuTorch + TensorRT FP4 (ET 1.4.1, host I/O) | 19.1 | 6.3 | 35.7 | 94 ms (68 ms fused) | 0.0196 |

What FP4 changes:

* **The prefix is now the cheap stage.** 26 → 16 → 8.8 ms from bf16 to FP8 to FP4 in TensorRT. Per
  6-layer chunk the layer profiler shows GEMM time 8.3 → 5.6 → 2.1 ms while the bf16 attention
  kernels stay at 0.7 ms, so a chunk goes 9.4 → 6.8 → 3.3 ms. Prefill of 968 tokens through 2.5 B
  parameters now costs less than the fp32 vision tower (9.7 ms).
* **The expert step does not benefit, and in FP4 it gets slower**: 3.0 (bf16) → 2.9 (FP8) → 3.4 ms
  (FP4). At 50 tokens the FP4 GEMM kernels take longer than the bf16 ones (profiled 1.2 → 1.5 →
  2.0 ms per step), the fused attention is unchanged at about 2.0 ms, and the block-scale arithmetic
  adds to the elementwise kernels (0.44 → 0.58 → 0.78 ms). The step runs 502 kernels against 210 in
  bf16. The fused ten-step engine follows: 26 → 31 ms. The cycle still improves, 62 → 53 → 51 ms
  (fused 50 → 47 ms), entirely from the prefix.
* **Eager FP4 is a bigger loss than eager FP8.** 540 ms per cycle: prefix 51 ms, expert step 46 ms.
  Without a fused quantiser kernel the per-block amax, scale and cast run as separate PyTorch ops on
  every linear input. Same lesson as FP8, louder.
* **Memory keeps halving where it can.** Expert engine 1.05 → 0.79 → 0.66 GB (the 453 MB of fp32
  AdaRMS weights now dominate it), each prefix chunk 1.26 → 0.66 → 0.37 GB, fused ten-step engine
  358 → 243 MB. The full engine set is 4.4 GB (4.0 GB with the fused expert) against 7.4 GB in bf16,
  and 2.6 GB of what remains is the unquantised vision tower and embedding table. Builds are quick
  (15 to 24 s per unit, 186 s for the fused engine) and the `.pte` serialisation stays in-process.
* **Accuracy cost is ten times bf16's.** Action MAE vs fp32 is 0.0196 (TensorRT) and 0.0209
  (torchao), against 0.0059 for FP8 and 0.0021 for bf16; max error 0.11 on actions of mean
  magnitude 0.32; gripper signs still agree on all 8 scenes. Modular and fused FP4 now differ by
  0.0036 (FP8: 1e-4), so ten compounded FP4 steps do drift. The error is the recipe, not the
  engine: the eager ModelOpt fake-quant pass lands at 0.0180 vs fp32, the engine sits 0.0118 from
  it, and the two eager FP4 implementations sit 0.0146 from each other. Whether 0.02 MAE is
  tolerable is a closed-loop question; the plain max-calibrated default is the weakest NVFP4 recipe,
  and ModelOpt's AWQ variants (`NVFP4_AWQ_LITE_CFG`) are the first thing to try before giving up on it.
* **The ExecuTorch + TensorRT numbers above were re-measured** in the same session as FP4. The
  host-copy path of that runtime is sensitive to the other tenant's CPU load: the earlier runs gave
  166 ms (bf16) and 141 ms (FP8) for the same engines that now measure 110 and 96 ms.

### Which one, then?

* **Lowest latency on an NVIDIA target: direct TensorRT**, 62 ms per cycle here, ~16 Hz of full
  re-planning for a policy that is typically re-planned every 5–10 steps. Pay for it in build
  memory, per-GPU engines, and a split model if the engine does not fit the host.
* **One runtime across GPU and CPU/NPU targets: ExecuTorch + CUDA.** It compiled the whole model
  including the 2.5 B prefix in one shot, keeps weights separate from the program, and gets the
  same expert-step speedup as TensorRT. On this GPU it gives back part of the gain on fp32 vision
  (bf16-only Triton attention) and, from Python, on I/O copies; the C++ runtime closes the latter.
* **ExecuTorch + TensorRT** gives TensorRT kernels inside the ExecuTorch program format, and on
  the same host-copy runtime it edges out ExecuTorch's own CUDA backend (110 vs 124 ms) on the
  strength of TensorRT's vision tower. But with the 2.14 release its Python runtime hides most of
  TensorRT's advantage behind host copies (110 ms against 62 ms from the raw runtime) and its `.pte`
  packaging is the most memory-hungry step of the whole study. It is the right shape for a C++
  ExecuTorch deployment that wants TensorRT on the GPU and ExecuTorch delegates elsewhere; it is not
  a shortcut to TensorRT performance from Python.

### Caveats

* One GPU, one model, batch 1, static shapes; the other tenant on the box adds CPU jitter (see p90).
  All table numbers come from one session (4 Nov 2026); the first session's runs are kept in `logs/`
  and differ mainly on the host-copy ExecuTorch + TensorRT path (see finding 3).
* The Python ExecuTorch runtimes understate what the C++ runtime would do (see finding 3).
* The fused 10-step TensorRT engine (and hence the ExecuTorch+TensorRT fused variant) could not be
  built in the ~11 GB of host memory available during this session; the TensorRT builder was
  OOM-killed on the 22.6k-node network.
* ExecuTorch 1.4.1's CUDA backend has no kernel for `aten::split_copy` (the AdaRMS `chunk(3)` in the
  expert); those units were exported with the chunk rewritten as three slices (`--split-to-slices`),
  which 1.5.1 does not need.
* Fidelity is measured on 8 LIBERO observations against the same noise sample, not in closed loop.

## Memory discipline

The box has 30 GB of RAM, a large unrelated process was using 12 GB of it during this work, and
LeRobot's stock loader would have needed ~22 GB (fp32 state dict next to the model). Two rules kept
everything alive:

* `pi05_edge/model.py` builds the policy *on the GPU* and streams the 14.5 GB safetensors in one tensor
  at a time; the host never holds more than the 2 GB embedding table.
* Every job runs under `scripts/run_limited.sh`: a transient systemd user cgroup with
  `MemoryMax = min(request, MemAvailable − 2 GB)`, no swap, `oom_score_adj=1000` so a global OOM would
  take the job and nothing else, and `HF_HUB_OFFLINE=1`. A failing export kills itself, never the box.

## Layout

```
scripts/setup_envs.sh     two uv environments: .venvs/trt (torch 2.14 · Torch-TensorRT 2.14 · TensorRT 11.1 ·
                          ExecuTorch 1.4.1 + torch-tensorrt-executorch-runtime) and .venvs/et15 (torch 2.15 ·
                          ExecuTorch 1.5.1+cu132)
scripts/run_limited.sh    cgroup wrapper (see above)
scripts/run_all.sh        the whole matrix, one GPU job at a time
pi05_edge/model.py        GPU-streaming loader, the four export units, the host-side glue
pi05_edge/data.py         LIBERO observations -> models/inputs/obs_bundle.pt
pi05_edge/bench_eager.py  PyTorch reference (fp32 / bf16), writes the reference action chunks
pi05_edge/export_et_cuda.py   path 1 export        pi05_edge/bench_et.py          runs any .pte (both ET runtimes)
pi05_edge/export_trt.py       paths 2+3 export     pi05_edge/bench_trt_direct.py  raw TensorRT runtime
pi05_edge/report.py       results/raw/*.json -> results/summary.md
```

## Reproduce

```bash
scripts/setup_envs.sh            # ~10 min, downloads ~4 GB of wheels
.venvs/et15/bin/python -m pi05_edge.data   # needs the LIBERO scenes from ../pi-0.7/vla_bench/data
scripts/run_all.sh               # hours; see logs/
```
