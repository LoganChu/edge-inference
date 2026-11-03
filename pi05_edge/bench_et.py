"""Run exported .pte programs (ExecuTorch+CUDA or ExecuTorch+TensorRT) and measure them.

Two Python runtimes are supported transparently:
  * executorch.runtime.Runtime            (ExecuTorch 1.5.x wheel from download.pytorch.org/whl/cu132)
  * torch_tensorrt_executorch_runtime     (Torch-TensorRT 2.14's bundled ExecuTorch 1.4.1 runtime with the
                                           TensorRT delegate *and* the AOTI CUDA backend linked in)

    python -m pi05_edge.bench_et --dir models/et_cuda --label et_cuda_1.5.1
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .common import MODELS, RESULTS, action_error, bench, mem_report, write_result
from .data import load_bundle, obs_inputs

METHODS = ["vision", "prefix", "prefix_embed", "prefix_l0", "prefix_l1", "prefix_l2", "denoise_step", "denoise10"]


class EtRuntime:
    """Uniform wrapper: load(dir/method) -> callable(*cuda tensors) -> cuda tensor(s)."""

    def __init__(self, runtime: str):
        self.runtime = runtime
        if runtime == "torch_tensorrt":
            import os
            from torch_tensorrt_executorch_runtime import activate, get_runtime
            get_runtime()
            self.native = activate()
            self.version = "executorch 1.4.1 (torch_tensorrt_executorch_runtime 0.1.0)"
            self.cpu_only = True  # this runtime's bindings read inputs as host memory; a CUDA pointer segfaults
        else:
            from executorch.runtime import Runtime, Verification
            self.rt = Runtime.get()
            self.Verification = Verification
            import importlib.metadata as md
            self.version = f"executorch {md.version('executorch')}"
            self.cpu_only = False

    def load(self, pte: Path, ptd: Path | None):
        if self.runtime == "torch_tensorrt":
            mod = self.native._load_for_executorch(str(pte), data_path=str(ptd) if ptd else None)
            names = list(mod.method_names())
            name = "forward" if "forward" in names else names[0]
            return lambda inputs: mod.run_method(name, inputs)
        prog = self.rt.load_program(pte, verification=self.Verification.Minimal, data_path=ptd)
        m = prog.load_method("forward")
        return lambda inputs: m.execute(inputs)


def make_callable(run, io: str):
    """io='cuda': hand CUDA tensors to the program.  io='cpu': copy inputs to the host first.
    Outputs are moved to the GPU if the program returned them on the host."""
    def fn(*tensors):
        if io == "cpu":
            outs = run([t.detach().cpu().contiguous() for t in tensors])
        else:
            outs = run([t.contiguous() for t in tensors])
        outs = [o if o.is_cuda else o.cuda() for o in outs]
        return outs[0] if len(outs) == 1 else tuple(outs)
    return fn


def detect_io(run, example: tuple, cpu_only: bool = False) -> str:
    """Try device-resident inputs first; fall back to host copies. Reports where outputs came back."""
    if cpu_only:
        outs = run([t.detach().cpu().contiguous() for t in example])
        return "in:cpu/out:" + ("cuda" if all(o.is_cuda for o in outs) else "cpu")
    try:
        outs = run([t.contiguous() for t in example])
        return "in:cuda/out:" + ("cuda" if all(o.is_cuda for o in outs) else "cpu")
    except Exception as e:  # noqa: BLE001
        print(f"   device-resident call failed ({type(e).__name__}: {str(e)[:120]}); using host I/O")
    outs = run([t.detach().cpu().contiguous() for t in example])
    return "in:cpu/out:" + ("cuda" if all(o.is_cuda for o in outs) else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory with <method>/<method>.pte (+ .ptd)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--runtime", default="auto", choices=["auto", "executorch", "torch_tensorrt"])
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--vision-only", action="store_true", help="only time the vision unit (precision-variant runs)")
    ap.add_argument("--ref", default="bfloat16", help="eager dtype whose actions are the fidelity reference")
    args = ap.parse_args()
    if args.runtime == "auto":
        try:
            import torch_tensorrt_executorch_runtime  # noqa: F401
            args.runtime = "torch_tensorrt"
        except ImportError:
            args.runtime = "executorch"

    mem_report("start")
    bundle = load_bundle()
    rt = EtRuntime(args.runtime)
    print(f"[runtime] {rt.version}")
    methods = args.methods.split(",")
    d = Path(args.dir)

    # shapes for the device-resident probe come from the export result files
    fns, io_modes, load_s, files = {}, {}, {}, {}
    inp0 = obs_inputs(bundle, 0)
    probe = {
        "vision": (inp0["images"],),
        "prefix": None, "denoise_step": None, "denoise10": None,
    }
    for m in methods:
        pte = d / m / f"{m}.pte"
        ptds = sorted((d / m).glob("*.ptd"))
        if not pte.exists():
            print(f"[skip] {pte} missing"); continue
        t0 = time.time()
        run = rt.load(pte, ptds[0] if ptds else None)
        load_s[m] = round(time.time() - t0, 2)
        files[m] = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted((d / m).iterdir())}
        fns[m] = run
        print(f"[load] {m}: {files[m]} MB in {load_s[m]}s")
    mem_report("programs loaded")

    # Chain probes so every method sees real tensors from the previous one.
    io_modes["vision"] = detect_io(fns["vision"], probe["vision"], rt.cpu_only)
    mode = lambda m: "cuda" if io_modes[m].startswith("in:cuda") else "cpu"  # noqa: E731
    vision = make_callable(fns["vision"], mode("vision"))
    img_tok = vision(inp0["images"])
    if args.vision_only:
        t = bench(lambda: vision(inp0["images"]), iters=args.iters)
        print(f"[time] vision {t['median_ms']:9.2f} ms (p90 {t['p90_ms']:.2f})")
        write_result(f"bench_{args.label}", {"path": args.label, "runtime": rt.version, "io_modes": io_modes, "files_MB": files,
                                             "timings_ms": {"vision": t}, "memory": mem_report("end")})
        return
    if "prefix" in fns:
        io_modes["prefix"] = detect_io(fns["prefix"], (img_tok, inp0["tokens"], inp0["pad_mask"]), rt.cpu_only)
        prefix = make_callable(fns["prefix"], mode("prefix"))
    else:
        from .model import Pi05Parts
        io_modes["prefix_embed"] = detect_io(fns["prefix_embed"], (img_tok, inp0["tokens"], inp0["pad_mask"]), rt.cpu_only)
        embed = make_callable(fns["prefix_embed"], mode("prefix_embed"))
        h, pos, mask = embed(img_tok, inp0["tokens"], inp0["pad_mask"])
        chunk_names = sorted(k for k in fns if k.startswith("prefix_l"))
        chunks = []
        for k in chunk_names:
            io_modes[k] = detect_io(fns[k], (h, pos, mask), rt.cpu_only)
            chunks.append(make_callable(fns[k], mode(k)))
        prefix = Pi05Parts.compose_prefix(embed, chunks)
        files["prefix"] = {"composed_from": ["prefix_embed"] + chunk_names}
        print(f"[compose] prefix = prefix_embed + {len(chunks)} layer chunks")
    k_all, v_all = prefix(img_tok, inp0["tokens"], inp0["pad_mask"])
    t1 = torch.ones(1, device="cuda")
    calls = {"vision": vision, "prefix": prefix}
    if "denoise_step" in fns:
        io_modes["denoise_step"] = detect_io(fns["denoise_step"], (k_all, v_all, inp0["pad_mask"], inp0["noise"], t1), rt.cpu_only)
        calls["denoise_step"] = make_callable(fns["denoise_step"], mode("denoise_step"))
    if "denoise10" in fns:
        io_modes["denoise10"] = detect_io(fns["denoise10"], (k_all, v_all, inp0["pad_mask"], inp0["noise"]), rt.cpu_only)
        calls["denoise10"] = make_callable(fns["denoise10"], mode("denoise10"))
    print(f"[io] {io_modes}")

    n_steps = 10

    def end_to_end(inp, fused):
        it = calls["vision"](inp["images"])
        k, v = calls["prefix"](it, inp["tokens"], inp["pad_mask"])
        if fused:
            return calls["denoise10"](k, v, inp["pad_mask"], inp["noise"])
        dt = -1.0 / n_steps
        x = inp["noise"]
        for s in range(n_steps):
            t = torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
            x = x + dt * calls["denoise_step"](k, v, inp["pad_mask"], x, t)
        return x

    # fidelity against the eager references
    refs = {}
    for dt_name in ["float32", "bfloat16"]:
        p = RESULTS / f"actions_eager_{dt_name}.pt"
        if p.exists():
            refs[dt_name] = torch.load(p)["actions"]
    fidelity = {}
    acts_mod, acts_fused = [], []
    for i in range(len(bundle["uids"])):
        inp = obs_inputs(bundle, i)
        if "denoise_step" in calls:
            acts_mod.append(end_to_end(inp, fused=False).cpu())
        if "denoise10" in calls:
            acts_fused.append(end_to_end(inp, fused=True).cpu())
    for variant, acts in [("modular", acts_mod), ("fused", acts_fused)]:
        if not acts:
            continue
        a = torch.cat(acts)
        torch.save({"uids": bundle["uids"], "actions": a}, RESULTS / f"actions_{args.label}_{variant}.pt")
        for dt_name, ref in refs.items():
            fidelity[f"{variant}_vs_eager_{dt_name}"] = action_error(a, ref[: a.shape[0]])
    print("[fidelity]", json.dumps(fidelity, indent=1))

    timings = {"vision": bench(lambda: calls["vision"](inp0["images"]), iters=args.iters),
               "prefix": bench(lambda: calls["prefix"](img_tok, inp0["tokens"], inp0["pad_mask"]), iters=args.iters)}
    if "denoise_step" in calls:
        timings["denoise_step"] = bench(lambda: calls["denoise_step"](k_all, v_all, inp0["pad_mask"], inp0["noise"], t1), iters=args.iters)
        timings["end_to_end_modular"] = bench(lambda: end_to_end(inp0, fused=False), iters=args.iters)
    if "denoise10" in calls:
        timings["denoise10"] = bench(lambda: calls["denoise10"](k_all, v_all, inp0["pad_mask"], inp0["noise"]), iters=args.iters)
        timings["end_to_end_fused"] = bench(lambda: end_to_end(inp0, fused=True), iters=args.iters)
    for k, v in timings.items():
        print(f"[time] {k:24s} {v['median_ms']:9.2f} ms (p90 {v['p90_ms']:.2f})")
    mem = mem_report("end")
    write_result(f"bench_{args.label}", {
        "path": args.label, "runtime": rt.version, "io_modes": io_modes, "load_s": load_s, "files_MB": files,
        "timings_ms": timings, "fidelity": fidelity, "memory": mem,
    })


if __name__ == "__main__":
    main()
