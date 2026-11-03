"""Path 2 — direct TensorRT: the serialized engines executed by the plain TensorRT runtime.

No PyTorch graph, no ExecuTorch: `tensorrt.Runtime` deserialises each .plan, one execution context
per engine, tensors bound by address, `execute_async_v3` on one CUDA stream.  torch is used only as
the CUDA allocator and to hold the observation tensors (what a C++ deployment would do with cudaMalloc).

    python -m pi05_edge.bench_trt_direct --dir models/trt --label trt_direct
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .common import RESULTS, action_error, bench, mem_report, write_result
from .data import load_bundle, obs_inputs

METHODS = ["vision", "prefix", "prefix_embed", "prefix_l0", "prefix_l1", "prefix_l2", "denoise_step", "denoise10"]


class TrtEngine:
    def __init__(self, plan: Path, io: dict, runtime, stream):
        import tensorrt as trt

        self.trt = trt
        t0 = time.time()
        self.engine = runtime.deserialize_cuda_engine(plan.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.load_s = round(time.time() - t0, 2)
        self.stream = stream
        self.in_names, self.out_names = io["inputs"], io["outputs"]
        self.arg_names = io.get("arg_names", io["inputs"])  # positional forward args -> engine bindings by name
        self.torch_dtype = {trt.bfloat16: torch.bfloat16, trt.float32: torch.float32, trt.float16: torch.float16,
                            trt.int64: torch.int64, trt.int32: torch.int32, trt.bool: torch.bool}
        self.outputs = []
        for n in self.out_names:
            shape = tuple(self.engine.get_tensor_shape(n))
            dt = self.torch_dtype[self.engine.get_tensor_dtype(n)]
            self.outputs.append(torch.empty(shape, dtype=dt, device="cuda"))
            self.ctx.set_tensor_address(n, self.outputs[-1].data_ptr())
        self.device_mem_mb = round(self.engine.device_memory_size_v2 / 1e6, 1)

    def __call__(self, *tensors):
        by_name = dict(zip(self.arg_names, tensors))
        self._keep = []
        for n in self.in_names:
            t = by_name[n].contiguous()
            expected = self.torch_dtype[self.engine.get_tensor_dtype(n)]
            if t.dtype != expected:
                t = t.to(expected)
            self._keep.append(t)  # keep alive until execute
            self.ctx.set_tensor_address(n, t.data_ptr())
        ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        assert ok, "execute_async_v3 failed"
        outs = [o.clone() for o in self.outputs] if len(self.outputs) > 1 else [self.outputs[0]]
        return outs[0] if len(outs) == 1 else tuple(outs)


def main():
    import tensorrt as trt

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", default="trt_direct")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--vision-only", action="store_true", help="only time the vision unit (precision-variant runs)")
    args = ap.parse_args()

    mem_report("start")
    bundle = load_bundle()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    stream = torch.cuda.current_stream()
    engines, files = {}, {}
    for m in args.methods.split(","):
        d = Path(args.dir) / m
        plan = d / f"{m}.plan"
        if not plan.exists():
            print(f"[skip] {plan} missing"); continue
        engines[m] = TrtEngine(plan, json.loads((d / "io.json").read_text()), runtime, stream)
        files[m] = {"plan_MB": round(plan.stat().st_size / 1e6, 1), "load_s": engines[m].load_s,
                    "activation_mem_MB": engines[m].device_mem_mb}
        print(f"[load] {m}: {files[m]}")
    mem_report("engines loaded")
    print(f"[trt] {trt.__version__}")
    if "prefix" not in engines and "prefix_embed" in engines:
        from .model import Pi05Parts
        chunks = [engines[k] for k in sorted(k for k in engines if k.startswith("prefix_l"))]
        engines["prefix"] = Pi05Parts.compose_prefix(engines["prefix_embed"], chunks)
        files["prefix"] = {"composed_from": ["prefix_embed"] + [f"prefix_l{i}" for i in range(len(chunks))]}
        print(f"[compose] prefix = prefix_embed + {len(chunks)} layer chunks")

    n_steps = 10
    inp0 = obs_inputs(bundle, 0)
    if args.vision_only:
        t = bench(lambda: engines["vision"](inp0["images"]), iters=args.iters)
        print(f"[time] vision {t['median_ms']:9.2f} ms (p90 {t['p90_ms']:.2f})")
        write_result(f"bench_{args.label}", {"path": args.label, "runtime": f"tensorrt {trt.__version__}", "files": files,
                                             "timings_ms": {"vision": t}, "memory": mem_report("end")})
        return

    def end_to_end(inp, fused):
        it = engines["vision"](inp["images"])
        k, v = engines["prefix"](it, inp["tokens"], inp["pad_mask"])
        if fused:
            return engines["denoise10"](k, v, inp["pad_mask"], inp["noise"])
        dt = -1.0 / n_steps
        x = inp["noise"]
        for s in range(n_steps):
            t = torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device="cuda")
            x = x + dt * engines["denoise_step"](k, v, inp["pad_mask"], x, t)
        return x

    refs = {dt: torch.load(RESULTS / f"actions_eager_{dt}.pt")["actions"]
            for dt in ["float32", "bfloat16"] if (RESULTS / f"actions_eager_{dt}.pt").exists()}
    fidelity = {}
    for variant, fused in [("modular", False), ("fused", True)]:
        need = "denoise10" if fused else "denoise_step"
        if need not in engines:
            continue
        acts = torch.cat([end_to_end(obs_inputs(bundle, i), fused).cpu() for i in range(len(bundle["uids"]))])
        torch.save({"uids": bundle["uids"], "actions": acts}, RESULTS / f"actions_{args.label}_{variant}.pt")
        for dt, ref in refs.items():
            fidelity[f"{variant}_vs_eager_{dt}"] = action_error(acts, ref[: acts.shape[0]])
    print("[fidelity]", json.dumps(fidelity, indent=1))

    img_tok = engines["vision"](inp0["images"])
    k_all, v_all = engines["prefix"](img_tok, inp0["tokens"], inp0["pad_mask"])
    t1 = torch.ones(1, device="cuda")
    timings = {"vision": bench(lambda: engines["vision"](inp0["images"]), iters=args.iters),
               "prefix": bench(lambda: engines["prefix"](img_tok, inp0["tokens"], inp0["pad_mask"]), iters=args.iters)}
    if "denoise_step" in engines:
        timings["denoise_step"] = bench(lambda: engines["denoise_step"](k_all, v_all, inp0["pad_mask"], inp0["noise"], t1), iters=args.iters)
        timings["end_to_end_modular"] = bench(lambda: end_to_end(inp0, False), iters=args.iters)
    if "denoise10" in engines:
        timings["denoise10"] = bench(lambda: engines["denoise10"](k_all, v_all, inp0["pad_mask"], inp0["noise"]), iters=args.iters)
        timings["end_to_end_fused"] = bench(lambda: end_to_end(inp0, True), iters=args.iters)
    for k, v in timings.items():
        print(f"[time] {k:24s} {v['median_ms']:9.2f} ms (p90 {v['p90_ms']:.2f})")
    mem = mem_report("end")
    write_result(f"bench_{args.label}", {"path": args.label, "runtime": f"tensorrt {trt.__version__} (python bindings, execute_async_v3)",
                                         "files": files, "timings_ms": timings, "fidelity": fidelity, "memory": mem})


if __name__ == "__main__":
    main()
