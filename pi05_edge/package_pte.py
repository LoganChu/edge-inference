"""Re-package an already built TensorRT engine (.plan + io.json) as an ExecuTorch .pte in a fresh process.

Torch-TensorRT 2.14 embeds the engine in the .pte through a base64 round trip, which needs a few
times the engine size in host RAM on top of whatever the exporting process still holds (the TensorRT
builder leaves the weights' host copies around). Doing it here, with nothing but the engine in memory,
keeps the peak at roughly baseline + 3x engine.

    python -m pi05_edge.package_pte --dir models/trt/vision
"""
import argparse
import json
import operator
import time
from pathlib import Path

import torch

from .common import mem_report, write_result

TORCH_DTYPE = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16, "torch.int64": torch.int64, "torch.bool": torch.bool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="models/trt/<method>")
    args = ap.parse_args()
    import torch_tensorrt
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
    from torch_tensorrt.dynamo._settings import CompilationSettings

    d = Path(args.dir)
    method = d.name
    io = json.loads((d / "io.json").read_text())
    mem_report("start")
    blob = (d / f"{method}.plan").read_bytes()
    example = [torch.zeros(s, dtype=TORCH_DTYPE[dt], device="cuda") for s, dt in zip(io["example_shapes"], io["example_dtypes"])]
    n_out = len(io["outputs"])
    # Output shapes/dtypes straight from the engine; static engines, so plain ints.
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = rt.deserialize_cuda_engine(blob)
    trt2torch = {trt.bfloat16: torch.bfloat16, trt.float32: torch.float32, trt.float16: torch.float16,
                 trt.int64: torch.int64, trt.int32: torch.int32, trt.bool: torch.bool}
    arg_names = io.get("arg_names", io["inputs"])
    symexprs = {
        "inputs": [{"shape_exprs": list(t.shape), "dtype": t.dtype, "name": n} for n, t in zip(arg_names, example)],
        "outputs": [{"shape_exprs": list(eng.get_tensor_shape(n)), "dtype": trt2torch[eng.get_tensor_dtype(n)]} for n in io["outputs"]],
    }
    del eng, rt
    trt_mod = TorchTensorRTModule(serialized_engine=blob, input_binding_names=io["inputs"],
                                  output_binding_names=io["outputs"], name="_run_on_acc_0",
                                  settings=CompilationSettings(device=torch_tensorrt.Device("cuda:0")),
                                  symbolic_shape_expressions=symexprs)

    # The same shape of GraphModule torch_tensorrt.dynamo.compile produces: placeholders -> call_module -> output
    root = torch.nn.Module()
    root._run_on_acc_0 = trt_mod
    graph = torch.fx.Graph()
    phs = [graph.placeholder(n) for n in arg_names]
    by_name = dict(zip(arg_names, phs))
    call = graph.call_module("_run_on_acc_0", args=tuple(by_name[n] for n in io["inputs"]))
    # Mirror dynamo.compile's graph: a single output is used directly (the exporter inserts the getitem
    # itself); several outputs are already split with getitem by the partitioner.
    if n_out == 1:
        graph.output((call,))
    else:
        graph.output(tuple(graph.call_function(operator.getitem, (call, i)) for i in range(n_out)))
    gm = torch.fx.GraphModule(root, graph)
    with torch.no_grad():
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*example)
        out = gm(*example)
    out = out if isinstance(out, (tuple, list)) else (out,)
    # The exporter wants the FakeTensor metadata that dynamo.compile leaves on the engine node.
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake_mode = FakeTensorMode()
    for ph, t in zip(phs, example):
        ph.meta["val"] = fake_mode.from_tensor(t)
    fakes = [fake_mode.from_tensor(o) for o in out]
    call.meta["val"] = fakes
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is operator.getitem:
            node.meta["val"] = fakes[node.args[1]]
    print(f"[engine] {method}: {len(blob)/1e6:.1f} MB, outputs {[tuple(o.shape) for o in out]}", flush=True)
    mem_report("engine loaded")

    t0 = time.time()
    torch_tensorrt.save(gm, str(d / f"{method}.pte"), output_format="executorch", arg_inputs=example, retrace=False)
    dt = time.time() - t0
    size = (d / f"{method}.pte").stat().st_size / 1e6
    print(f"[pte] wrote {method}.pte ({size:.1f} MB) in {dt:.1f}s", flush=True)
    mem = mem_report("end")
    write_result(f"package_pte_{method}", {"method": method, "pte_MB": round(size, 1), "pte_save_s": round(dt, 1), "memory": mem})


if __name__ == "__main__":
    main()
