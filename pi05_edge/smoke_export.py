"""torch.export every unit once (no backend lowering) and report node counts + the aten op set."""
import argparse
import collections
import time

import torch

from .common import mem_report, write_result
from .model import Pi05Parts, load_pi05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="vision,prefix,denoise_step,denoise10")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    policy = load_pi05(dtype=args.dtype)
    parts = Pi05Parts(policy)
    ex = parts.example_inputs()
    out = {}
    for m in args.methods.split(","):
        t0 = time.time()
        try:
            with torch.no_grad():
                ep = torch.export.export(getattr(parts, m), ex[m], strict=False)
            ops = collections.Counter(str(n.target) for n in ep.graph.nodes if n.op == "call_function")
            out[m] = {"ok": True, "s": round(time.time() - t0, 1), "nodes": len(ep.graph.nodes),
                      "n_params": len(ep.graph_signature.parameters), "n_buffers": len(ep.graph_signature.buffers),
                      "ops": dict(ops.most_common())}
            print(f"[ok] {m}: {len(ep.graph.nodes)} nodes, {len(ops)} distinct ops, {out[m]['s']}s")
            print("     top ops:", ops.most_common(12))
            del ep
        except Exception as e:  # noqa: BLE001
            out[m] = {"ok": False, "s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {str(e)[:2000]}"}
            print(f"[FAIL] {m}: {type(e).__name__}: {str(e)[:1500]}")
        mem_report(m)
    write_result("smoke_export", {"dtype": args.dtype, "methods": out})


if __name__ == "__main__":
    main()
