"""Paths 2 and 3 — TensorRT via Torch-TensorRT's dynamo front end.

The same torch.export graph is converted once into a TensorRT engine
(`torch_tensorrt.dynamo.compile`, full compilation, strongly typed so bf16/f32 follow the model).
That one engine is then packaged two ways:

  * direct TensorRT     <out>/<method>/<method>.plan  + io.json   -> run by the raw TensorRT runtime
  * ExecuTorch+TensorRT <out>/<method>/<method>.pte                -> run by the ExecuTorch runtime
                                                                       with the TensorRT delegate

    python -m pi05_edge.export_trt --method vision --out models/trt
"""
import argparse
import inspect
import json
import time
from pathlib import Path

import torch

from .common import MODELS, mem_report, write_result
from .model import Pi05Parts, load_pi05

TRT_DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def trt_settings(method: str, timing_cache: Path, disable_tf32: bool = False) -> dict:
    import torch_tensorrt

    want = dict(
        disable_tf32=disable_tf32,
        min_block_size=1,
        require_full_compilation=True,     # one engine, nothing left to PyTorch
        truncate_double=True,              # sinusoidal time embedding is computed in f64
        timing_cache_path=str(timing_cache),
        immutable_weights=True,
        deduplicate_weights=True,          # denoise10 reuses every expert weight 10x
        use_fp32_acc=False,
    )
    accepted = inspect.signature(torch_tensorrt.dynamo.compile).parameters
    dropped = [k for k in want if k not in accepted]
    return {k: v for k, v in want.items() if k in accepted}, dropped


def extract_engines(trt_gm):
    """All TensorRT engines inside a compiled GraphModule -> [(name, bytes, in_names, out_names)]."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    engines = []
    for name, mod in trt_gm.named_children():
        if isinstance(mod, TorchTensorRTModule):
            engines.append((name, mod.serialized_engine, list(mod.input_binding_names), list(mod.output_binding_names)))
        elif hasattr(mod, "engine") and hasattr(mod.engine, "serialize"):  # python runtime module
            engines.append((name, bytes(mod.engine.serialize()), list(mod.input_names), list(mod.output_names)))
    return engines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="vision | prefix | prefix_embed | prefix_l<i> | denoise_step | denoise10")
    ap.add_argument("--prefix-chunks", type=int, default=3)
    ap.add_argument("--out", default=str(MODELS / "trt"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--skip-pte", action="store_true", help="only write the raw .plan")
    ap.add_argument("--skip-plan", action="store_true")
    ap.add_argument("--dryrun", action="store_true", help="partition only; report which ops TensorRT will not take")
    ap.add_argument("--disable-tf32", action="store_true", help="fp32 layers in true fp32 instead of TensorRT's default TF32")
    ap.add_argument("--quant", default="none", choices=["none", "fp8", "fp4"],
                    help="ModelOpt PTQ on the Gemma linears, calibrated on models/inputs/calib_bundle.pt: fp8 = W8A8 E4M3 per-tensor static; fp4 = NVFP4 W4A4 block-16")
    ap.add_argument("--n-calib", type=int, default=40)
    args = ap.parse_args()

    import torch_tensorrt
    import tensorrt as trt

    out_dir = Path(args.out) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    mem_report("start")
    policy = load_pi05(dtype=args.dtype)
    parts = Pi05Parts(policy, prefix_chunks=args.prefix_chunks)
    module = parts.unit(args.method)
    example = parts.example_inputs()[args.method]
    mem_report("model loaded")
    quant_info = {"recipe": "none"}
    if args.quant != "none":
        from .data import load_bundle
        from .quantize import modelopt_quantize
        quant_info = modelopt_quantize(args.quant, policy, parts, load_bundle(path=MODELS / "inputs" / "calib_bundle.pt"), args.n_calib)
        print(f"[quant] {quant_info}", flush=True)
        mem_report("after calibration")

    from .quantize import export_context
    t0 = time.time()
    with torch.no_grad(), export_context(f"{args.quant}_modelopt" if args.quant != "none" else "none"):
        ep = torch.export.export(module, example, strict=False)
    t_export = time.time() - t0
    print(f"[export] torch.export {args.method}: {len(ep.graph.nodes)} nodes in {t_export:.1f}s", flush=True)
    from .graph_passes import strip_empty_cats
    with torch.no_grad():
        before = module(*example)
        n_removed = strip_empty_cats(ep)
        after = ep.module()(*example)
    before = before if isinstance(before, (tuple, list)) else (before,)
    after = after if isinstance(after, (tuple, list)) else (after,)
    pass_diff = max((a.float() - b.float()).abs().max().item() for a, b in zip(after, before))
    print(f"[pass] strip_empty_cats removed {n_removed} cat nodes; graph now {len(ep.graph.nodes)} nodes; max abs diff vs eager {pass_diff:.3e}", flush=True)
    del before, after

    if args.dryrun:
        import logging
        logging.getLogger("torch_tensorrt").setLevel(logging.DEBUG)
        from torch_tensorrt.dynamo.conversion import DYNAMO_CONVERTERS
        from torch_tensorrt.dynamo.lowering import get_decompositions
        gm = torch_tensorrt.dynamo.compile(ep, arg_inputs=list(example), min_block_size=1, truncate_double=True,
                                           dryrun=True, require_full_compilation=False)
        import collections
        unsupported = collections.Counter()
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target not in DYNAMO_CONVERTERS and "_run_on_acc" not in str(node.target):
                unsupported[str(node.target)] += 1
        print("[dryrun] call_function nodes left outside TensorRT:", dict(unsupported))
        print("[dryrun] submodules:", [n for n, _ in gm.named_children()])
        return

    settings, dropped = trt_settings(args.method, Path(args.out) / "timing_cache.bin", args.disable_tf32)
    if dropped:
        print(f"[trt] settings not supported by this Torch-TensorRT, dropped: {dropped}")
    t0 = time.time()
    with torch.no_grad():
        trt_gm = torch_tensorrt.dynamo.compile(ep, arg_inputs=list(example), **settings)
    t_build = time.time() - t0
    engines = extract_engines(trt_gm)
    print(f"[trt] built {len(engines)} engine(s) in {t_build:.1f}s: "
          f"{[(n, round(len(b)/1e6,1)) for n, b, *_ in engines]} MB", flush=True)
    mem_report("after TRT build")
    with torch.no_grad():
        ref = module(*example)
        got = trt_gm(*example)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    got = got if isinstance(got, (tuple, list)) else (got,)
    max_abs = max((r.float() - g.float()).abs().max().item() for r, g in zip(ref, got))
    ref_max = max(r.float().abs().max().item() for r in ref)
    rel_mae = sum((r.float() - g.float()).abs().mean().item() for r, g in zip(ref, got)) / max(
        1e-12, sum(r.float().abs().mean().item() for r in ref))
    print(f"[check] in-process TRT vs eager on zero inputs: max abs diff {max_abs:.3e} (ref max {ref_max:.3e}), relative MAE {rel_mae:.3e}")
    del ref, got

    result = {
        "path": "tensorrt", "method": args.method, "dtype": args.dtype, "quant": quant_info,
        "torch_tensorrt": torch_tensorrt.__version__, "tensorrt": trt.__version__,
        "export_s": round(t_export, 1), "trt_build_s": round(t_build, 1), "settings": {k: str(v) for k, v in settings.items()},
        "engines": [{"name": n, "MB": round(len(b) / 1e6, 1), "inputs": i, "outputs": o} for n, b, i, o in engines],
        "zero_input_max_abs_diff": max_abs, "zero_input_ref_max": ref_max, "zero_input_relative_mae": rel_mae, "graph_nodes": len(ep.graph.nodes), "graph_passes": {"strip_empty_cats_removed": n_removed},
    }

    if not args.skip_plan:
        if len(engines) != 1:
            print(f"[plan] WARNING: {len(engines)} engines; direct-TRT path needs exactly one — writing the first")
        name, blob, in_names, out_names = engines[0]
        (out_dir / f"{args.method}.plan").write_bytes(blob)
        arg_names = list(inspect.signature(module.forward).parameters)
        io = {"inputs": in_names, "outputs": out_names, "arg_names": arg_names,
              "example_shapes": [list(t.shape) for t in example], "example_dtypes": [str(t.dtype) for t in example]}
        (out_dir / "io.json").write_text(json.dumps(io, indent=1))
        print(f"[plan] wrote {args.method}.plan ({len(blob)/1e6:.1f} MB)", flush=True)
        result["files_MB"] = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(out_dir.iterdir())}
        result["memory"] = mem_report("after .plan")
        write_result(f"export_trt_{args.method}" + (f"_{args.quant}" if args.quant != "none" else ""), result)  # partial record in case the .pte step is OOM-killed

    if not args.skip_pte:
        import gc
        del policy, parts, module, ep
        gc.collect(); torch.cuda.empty_cache()
        mem_report("model released before .pte save")
        t0 = time.time()
        try:
            torch_tensorrt.save(trt_gm, str(out_dir / f"{args.method}.pte"), output_format="executorch",
                                arg_inputs=list(example), retrace=False)
            result["pte_save_s"] = round(time.time() - t0, 1)
            print(f"[pte] wrote {args.method}.pte in {result['pte_save_s']}s", flush=True)
        except Exception as e:  # noqa: BLE001
            result["pte_error"] = f"{type(e).__name__}: {str(e)[:500]}"
            print(f"[pte] FAILED after {time.time()-t0:.1f}s: {result['pte_error']}", flush=True)
        mem_report("after .pte save")

    result["files_MB"] = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(out_dir.iterdir())}
    result["memory"] = mem_report("end")
    write_result(f"export_trt_{args.method}" + (f"_{args.quant}" if args.quant != "none" else ""), result)


if __name__ == "__main__":
    main()
