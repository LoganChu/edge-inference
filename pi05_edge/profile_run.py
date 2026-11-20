"""The control cycle as a profiler workload: NVTX-labelled stages inside a cudaProfilerStart/Stop window.

Model loading (14.5 GB streamed into the GPU, thousands of copies) and warm-up stay outside the
capture window, so `nsys --capture-range=cudaProfilerApi` and `ncu --profile-from-start off` see only
the steady-state cycle a robot controller would run:

    cycle            NVTX range around one full control cycle
      vision         SigLIP + projector (3 cameras)
      prefix         Gemma-2B prefill -> KV cache
      denoise_step   one Euler step of the Gemma-300M expert (x10)

The glue between stages (the Euler update, the timestep tensor) is the same host code as
`Pi05Parts.run_modular` / `bench_trt_direct.end_to_end`, so a profiled cycle is a benchmarked cycle.

    python -m pi05_edge.profile_run --path eager                       # PyTorch eager bf16
    python -m pi05_edge.profile_run --path trt --dir models/trt        # direct TensorRT runtime
    python -m pi05_edge.profile_run --path eager --cycles 1 --steps 1  # ncu: every distinct kernel once
"""
import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .data import load_bundle, obs_inputs


@contextmanager
def nvtx(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def eager_fns(dtype: str):
    from .model import Pi05Parts, load_pi05

    parts = Pi05Parts(load_pi05(dtype=dtype))
    return {"vision": parts.vision, "prefix": parts.prefix, "denoise_step": parts.denoise_step}, parts.cfg.num_inference_steps


def trt_fns(d: Path):
    import tensorrt as trt

    from .bench_trt_direct import TrtEngine
    from .model import Pi05Parts

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    stream = torch.cuda.current_stream()
    eng = {}
    for m in ["vision", "prefix", "prefix_embed", "prefix_l0", "prefix_l1", "prefix_l2", "denoise_step"]:
        plan = d / m / f"{m}.plan"
        if plan.exists():
            eng[m] = TrtEngine(plan, json.loads((d / m / "io.json").read_text()), runtime, stream)
    if "prefix" not in eng:
        eng["prefix"] = Pi05Parts.compose_prefix(eng["prefix_embed"], [eng[k] for k in sorted(k for k in eng if k.startswith("prefix_l"))])
    print(f"[trt] {trt.__version__}: {sorted(eng)}", flush=True)
    return {"vision": eng["vision"], "prefix": eng["prefix"], "denoise_step": eng["denoise_step"]}, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, choices=["eager", "trt"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="eager only")
    ap.add_argument("--dir", default="models/trt", help="trt only: engine directory")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=10, help="profiled control cycles")
    ap.add_argument("--steps", type=int, default=None, help="denoising steps per profiled cycle (default: the policy's 10)")
    args = ap.parse_args()

    bundle = load_bundle()
    inp = obs_inputs(bundle, 0)
    fns, n_steps = eager_fns(args.dtype) if args.path == "eager" else trt_fns(Path(args.dir))
    dt = -1.0 / n_steps
    prof_steps = args.steps or n_steps

    def cycle(steps: int):
        with nvtx("cycle"):
            with nvtx("vision"):
                img_tok = fns["vision"](inp["images"])
            with nvtx("prefix"):
                k_all, v_all = fns["prefix"](img_tok, inp["tokens"], inp["pad_mask"])
            x = inp["noise"]
            for s in range(steps):
                with nvtx("denoise_step"):
                    t = torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
                    x = x + dt * fns["denoise_step"](k_all, v_all, inp["pad_mask"], x, t)
        return x

    with torch.inference_mode():
        for _ in range(args.warmup):
            cycle(n_steps)
        torch.cuda.synchronize()
        wall = []
        torch.cuda.profiler.start()
        for _ in range(args.cycles):
            t0 = time.perf_counter()
            cycle(prof_steps)
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.profiler.stop()
    wall.sort()
    print(f"[profile] path={args.path} cycles={args.cycles} steps={prof_steps} "
          f"wall median {wall[len(wall)//2]:.2f} ms (min {wall[0]:.2f}, max {wall[-1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
