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
| **ExecuTorch + CUDA** (AOTInductor, ET 1.5.1, Python runtime) | 33.0 | 27.5 | **2.9** | **101 ms** | 1.9× |
| **Direct TensorRT** (11.1, raw runtime) | **9.8** | 27.2 | **2.9** | **62 ms** | 3.0× |
| **ExecuTorch + TensorRT** (delegate, ET 1.4.1, Python runtime) | 12.2 | 89.3 | 8.1 | 166 ms | 1.1× |
| ExecuTorch + CUDA on the same ET 1.4.1 runtime, host I/O | 33.1 | 27.3 | 5.4 | 120 ms (91 ms with the fused 10-step unit) | 1.6× / 2.1× |
| ExecuTorch + CUDA, ET 1.4.1, prefix split in the same 4 chunks as TensorRT | 32.6 | 38.2 | 5.5 | 129 ms (104 ms fused) | 1.5× / 1.8× |

All compiled paths reproduce the bf16 reference to bf16 noise (action MAE ≈ 0.0009 vs eager bf16,
≈ 0.002 vs eager fp32, identical gripper signs); the fp32 eager run reproduces LeRobot's own loop
exactly (MAE 0).

### What the numbers say

**1. The win is in the action expert, and all compilers get it.** The Gemma-300M expert step is
13 ms in eager PyTorch and 2.9 ms under both AOTInductor and TensorRT: ten steps go from 134 ms to
29 ms. Eager PyTorch is launch-bound here (18 layers × dozens of small kernels on a 50-token
sequence), and fusion plus static shapes remove that. The 2.5 B-parameter prefix prefill is
GEMM-bound and already runs at tensor-core speed in eager bf16, so every path lands at 27–30 ms.
Fusing the ten steps into one graph (`denoise10`) buys nothing more once the step itself is
compiled: 27.9 ms for ten fused steps vs 29.0 ms for ten calls.

**2. "fp32" means different things.** TensorRT computes fp32 layers in TF32 by default; PyTorch
and AOTInductor do not. That alone is the 29 → 9.8 ms vision gap: eager PyTorch with TF32 enabled
runs the same SigLIP tower in 16 ms, and the action error is unchanged (MAE 0.0021 either way).
The remaining 16 → 9.8 ms is TensorRT's attention/GEMM fusion. ExecuTorch's CUDA backend was
*slower* than eager on the fp32 vision tower (33 ms) because its Triton attention kernel is
bf16-only and the fp32 tower falls back to a decomposed attention. The TF32-controlled variants
isolate the two effects on the vision tower alone:

| SigLIP + projector, 3 images | true fp32 | TF32 |
|---|---|---|
| PyTorch eager | 29.1 ms | 16.0 ms |
| ExecuTorch + CUDA (AOTInductor) | 33.0 ms | 13.0 ms |
| Direct TensorRT | 14.3 ms | 9.8 ms |

At equal precision TensorRT is 2× faster than eager on this tower and AOTInductor is on par with or
slightly ahead of eager; the headline 3× vision gap in the main table is two-thirds precision.

**3. The runtime boundary decides whether a delegate's speed reaches the robot.** The same
TensorRT engines that run a cycle in 62 ms from the raw runtime take 166 ms through the
ExecuTorch 1.4.1 Python runtime that Torch-TensorRT 2.14 ships, because that runtime copies every
input to the host and every output back (expert step 2.9 → 8.1 ms: the 18 MB KV cache crosses
PCIe twice per step). Running ExecuTorch+CUDA through the *same* 1.4.1 runtime with the same host
copies separates the two costs: the boundary itself turns the expert step into 5.4 ms and a
cycle into 120 ms, and splitting the prefix into the four chunks TensorRT needed costs another
11 ms (27 → 38 ms); the TensorRT delegate's own per-call handling accounts for the rest
(38 → 89 ms on the prefix, 5.4 → 8.1 ms on the step). On a host-copy runtime the fused 10-step
unit finally pays off, because the KV cache is uploaded once instead of ten times (120 → 91 ms).
ExecuTorch 1.5.1's runtime can take device-resident *inputs* but its Python bindings still
allocate outputs on the host; fully device-resident I/O is a C++-only fast path (the `Module`
API, shipped as a C++ SDK in the 1.5.1 wheel). The ET+CUDA figure of 101 ms therefore still
carries ~10 ms of host round trips a C++ deployment would not pay. Direct TensorRT has no such
boundary: engines are bound to device pointers and launched on one stream.

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

### Which one, then?

* **Lowest latency on an NVIDIA target: direct TensorRT**, 62 ms per cycle here, ~16 Hz of full
  re-planning for a policy that is typically re-planned every 5–10 steps. Pay for it in build
  memory, per-GPU engines, and a split model if the engine does not fit the host.
* **One runtime across GPU and CPU/NPU targets: ExecuTorch + CUDA.** It compiled the whole model
  including the 2.5 B prefix in one shot, keeps weights separate from the program, and gets the
  same expert-step speedup as TensorRT. On this GPU it gives back part of the gain on fp32 vision
  (bf16-only Triton attention) and, from Python, on I/O copies; the C++ runtime closes the latter.
* **ExecuTorch + TensorRT** gives TensorRT kernels inside the ExecuTorch program format, but with
  the 2.14 release its Python runtime hides TensorRT's advantage behind host copies (it ends up
  slower than ExecuTorch's own CUDA backend on the same runtime: 166 vs 120 ms) and its `.pte`
  packaging is the most memory-hungry step of the whole study. It is the right shape for a C++
  ExecuTorch deployment that wants TensorRT on the GPU and ExecuTorch delegates elsewhere; it is not
  a shortcut to TensorRT performance from Python.

### Caveats

* One GPU, one model, batch 1, static shapes; the other tenant on the box adds CPU jitter (see p90).
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
