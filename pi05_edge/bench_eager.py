"""PyTorch eager reference: ground-truth actions and the latency every deployment path is judged against.

    python -m pi05_edge.bench_eager --dtype float32     # numerics reference (14.5 GB of weights)
    python -m pi05_edge.bench_eager --dtype bfloat16    # deployment precision, same code path
"""
import argparse
import time

import torch

from .common import RESULTS, action_error, bench, mem_report, write_result
from .data import load_bundle, obs_inputs
from .model import Pi05Parts, load_pi05, param_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n-obs", type=int, default=None, help="fidelity on this many observations (default all)")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 tensor-core math for fp32 GEMMs/convs (TensorRT's default)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # default: keep the fp32 reference honest
    torch.backends.cudnn.allow_tf32 = args.tf32
    label = f"eager_{args.dtype}" + ("_tf32" if args.tf32 else "")

    mem_report("start")
    bundle = load_bundle()
    policy = load_pi05(dtype=args.dtype)
    parts = Pi05Parts(policy)
    summary = param_summary(policy)
    print("[params]", summary)
    mem_report("model loaded")

    n_obs = args.n_obs or len(bundle["uids"])
    ref_actions, mod_err, fused_err = [], [], []
    t0 = time.time()
    for i in range(n_obs):
        inp = obs_inputs(bundle, i)
        ref = parts.run_reference(inp)
        mod = parts.run_modular(inp, fused=False)
        fused = parts.run_modular(inp, fused=True)
        ref_actions.append(ref.cpu())
        mod_err.append(action_error(mod, ref)["mae"])
        fused_err.append(action_error(fused, ref)["mae"])
    ref_actions = torch.cat(ref_actions)
    print(f"[fidelity] modular-vs-reference MAE: max {max(mod_err):.3e}; fused: max {max(fused_err):.3e} "
          f"({n_obs} obs, {time.time()-t0:.1f}s)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save({"uids": bundle["uids"][:n_obs], "actions": ref_actions}, RESULTS / f"actions_{label}.pt")

    inp = obs_inputs(bundle, 0)
    ex = parts.example_inputs()
    with torch.no_grad():
        img_tok = parts.vision(inp["images"])
        k_all, v_all = parts.prefix(img_tok, inp["tokens"], inp["pad_mask"])
        t1 = torch.ones(1, device="cuda")
        timings = {
            "end_to_end_reference": bench(lambda: parts.run_reference(inp), iters=args.iters),
            "end_to_end_modular": bench(lambda: parts.run_modular(inp), iters=args.iters),
            "end_to_end_fused": bench(lambda: parts.run_modular(inp, fused=True), iters=args.iters),
            "vision": bench(lambda: parts.vision(inp["images"]), iters=args.iters),
            "prefix": bench(lambda: parts.prefix(img_tok, inp["tokens"], inp["pad_mask"]), iters=args.iters),
            "denoise_step": bench(lambda: parts.denoise_step(k_all, v_all, inp["pad_mask"], inp["noise"], t1), iters=args.iters),
            "denoise10": bench(lambda: parts.denoise10(k_all, v_all, inp["pad_mask"], inp["noise"]), iters=args.iters),
        }
    for k, v in timings.items():
        print(f"[time] {k:24s} {v['median_ms']:9.2f} ms (p90 {v['p90_ms']:.2f})")
    mem = mem_report("end")
    write_result(label, {
        "path": "pytorch_eager", "dtype": args.dtype, "tf32": args.tf32, "params": summary, "timings_ms": timings,
        "fidelity_modular_vs_reference_mae_max": max(mod_err), "fidelity_fused_vs_reference_mae_max": max(fused_err),
        "shapes": {k: [tuple(t.shape) for t in v] for k, v in ex.items()}, "memory": mem,
    })


if __name__ == "__main__":
    main()
