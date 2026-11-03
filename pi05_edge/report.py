"""Collect results/raw/*.json into results/summary.md."""
import json
from pathlib import Path

from .common import RESULTS

OUT = RESULTS.parent / "summary.md"
METHODS = ["vision", "prefix", "denoise_step", "denoise10", "end_to_end_modular", "end_to_end_fused"]
PATH_ORDER = ["eager_float32", "eager_bfloat16", "eager_bfloat16_tf32", "et_cuda_1.5.1", "et_cuda_1.4.1", "et_cuda_1.4.1_chunked",
              "trt_direct", "et_trt_1.4.1", "et_cuda_1.5.1_vision_tf32", "trt_direct_vision_fp32"]
PATH_LABEL = {
    "eager_float32": "PyTorch eager fp32 (reference)",
    "eager_bfloat16": "PyTorch eager bf16",
    "eager_bfloat16_tf32": "PyTorch eager bf16, TF32 for fp32 layers",
    "et_cuda_1.5.1_vision_tf32": "ExecuTorch + CUDA, vision compiled with TF32",
    "trt_direct_vision_fp32": "Direct TensorRT, vision with TF32 disabled",
    "et_cuda_1.5.1": "ExecuTorch + CUDA (AOTInductor), ET 1.5.1",
    "et_cuda_1.4.1": "ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O",
    "et_cuda_1.4.1_chunked": "ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT",
    "trt_direct": "Direct TensorRT 11.1 runtime",
    "et_trt_1.4.1": "ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks",
}


def load():
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}


def fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    R = load()
    bench = {}
    for k, v in R.items():
        if k.startswith("eager_"):
            bench[k] = v
        elif k.startswith("bench_"):
            bench[k[len("bench_"):]] = v
    paths = [p for p in PATH_ORDER if p in bench] + [p for p in bench if p not in PATH_ORDER]

    lines = ["# π₀.5 edge deployment paths — measured on one RTX 5090", ""]
    lines += ["## Latency (median ms, batch 1, 3 cameras × 224², 200 text tokens, 10 flow-matching steps)", ""]
    lines.append("| path | " + " | ".join(METHODS) + " |")
    lines.append("|---|" + "---|" * len(METHODS))
    for p in paths:
        t = bench[p].get("timings_ms", {})
        if p.startswith("eager_"):
            t = dict(t)
            t.setdefault("end_to_end_modular", t.get("end_to_end_modular"))
        row = [fmt(t[m]["median_ms"], 2) if m in t else "—" for m in METHODS]
        lines.append(f"| {PATH_LABEL.get(p, p)} | " + " | ".join(row) + " |")
    if "eager_float32" in bench:
        t = bench["eager_float32"]["timings_ms"]
        lines += ["", f"LeRobot's own `sample_actions` (fp32): {fmt(t['end_to_end_reference']['median_ms'], 1)} ms; "
                  f"bf16: {fmt(bench.get('eager_bfloat16', {}).get('timings_ms', {}).get('end_to_end_reference', {}).get('median_ms'), 1)} ms."]

    # Fidelity is recomputed here from the saved action chunks, so every path is compared against the
    # same references regardless of the order the benchmarks ran in.
    import torch
    from .common import action_error
    acts = {p.stem[len("actions_"):]: torch.load(p)["actions"] for p in RESULTS.glob("actions_*.pt")}
    ref32, ref16 = acts.get("eager_float32"), acts.get("eager_bfloat16")
    lines += ["", "## Fidelity of the predicted action chunk (MAE on the 7 real action dims, 8 LIBERO observations, same noise)", ""]
    lines.append("| path | variant | MAE vs eager fp32 | max abs vs fp32 | MAE vs eager bf16 | gripper sign agreement vs fp32 |")
    lines.append("|---|---|---|---|---|---|")
    rows = [("eager_bfloat16", "", "eager_bfloat16"), ("eager_bfloat16_tf32", "", "eager_bfloat16_tf32")]
    for p in paths:
        for variant in ["modular", "fused"]:
            if f"{p}_{variant}" in acts:
                rows.append((p, variant, f"{p}_{variant}"))
    for p, variant, key in rows:
        a = acts.get(key)
        if a is None:
            continue
        e32 = action_error(a, ref32[: a.shape[0]]) if ref32 is not None and key != "eager_float32" else None
        e16 = action_error(a, ref16[: a.shape[0]]) if ref16 is not None and key != "eager_bfloat16" else None
        lines.append(f"| {PATH_LABEL.get(p, p)} | {variant or '—'} | {fmt(e32 and e32['mae'], 4)} | {fmt(e32 and e32['max_abs'], 3)} | "
                     f"{fmt(e16 and e16['mae'], 4)} | {fmt(e32 and e32['gripper_sign_agree'], 3)} |")
    if "eager_bfloat16" in bench and "fidelity_modular_vs_reference_mae_max" in bench["eager_bfloat16"]:
        lines.append(f"| PyTorch eager bf16 | export units vs LeRobot sample_actions (same process) | MAE {bench['eager_bfloat16']['fidelity_modular_vs_reference_mae_max']:.2e} | | | |")
    if "eager_float32" in bench and "fidelity_modular_vs_reference_mae_max" in bench["eager_float32"]:
        lines.append(f"| PyTorch eager fp32 | export units vs LeRobot sample_actions (same process) | MAE {bench['eager_float32']['fidelity_modular_vs_reference_mae_max']:.2e} | | | |")

    lines += ["", "## Export / build cost and artifacts", ""]
    lines.append("| path | method | torch.export s | backend build s | artifact MB | peak host RSS GB |")
    lines.append("|---|---|---|---|---|---|")
    for k, v in R.items():
        if not (k.startswith("export_") or k.startswith("package_pte_")):
            continue
        if k.startswith("package_pte_"):
            m = v.get("memory", {})
            lines.append(f"| tensorrt (.pte re-packaged in a fresh process) | {v.get('method')} | — | {fmt(v.get('pte_save_s'))} | {fmt(v.get('pte_MB'))} (.pte) | {fmt(m.get('cgroup_peak_gb') or m.get('peak_rss_gb'), 2)} |")
            continue
        files = v.get("files_MB", {})
        tot = sum(files.values())
        build = v.get("lower_s", v.get("trt_build_s"))
        peak = v.get("memory", {}).get("cgroup_peak_gb") or v.get("memory", {}).get("peak_rss_gb")
        ver = v.get("executorch") or (f"torch-tensorrt {v.get('torch_tensorrt')}" if v.get("torch_tensorrt") else "")
        extra = " (TF32)" if v.get("tf32") else ""
        lines.append(f"| {v.get('path')} {ver}{extra} | {v.get('method')} | {fmt(v.get('export_s'))} | {fmt(build)} | {fmt(tot)} ({', '.join(f'{n} {s}' for n, s in files.items())}) | {fmt(peak, 2)} |")

    lines += ["", "## Runtime memory", ""]
    lines.append("| path | host RSS GB at end | peak host GB | GPU used by process GB (torch allocator) | device used GB |")
    lines.append("|---|---|---|---|---|")
    for p in paths:
        m = bench[p].get("memory", {})
        lines.append(f"| {PATH_LABEL.get(p, p)} | {fmt(m.get('rss_gb'), 2)} | {fmt(m.get('cgroup_peak_gb') or m.get('peak_rss_gb'), 2)} | "
                     f"{fmt(m.get('torch_max_allocated_gb'), 2)} | {fmt(m.get('device_used_gb'), 2)} |")
    io = {p: bench[p].get("io_modes") for p in paths if bench[p].get("io_modes")}
    if io:
        lines += ["", "I/O placement at the program boundary (cuda = device-resident, cpu = copied through host each call):", ""]
        for p, modes in io.items():
            lines.append(f"- {PATH_LABEL.get(p, p)}: {modes}")
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
