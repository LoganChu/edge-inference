"""Path 1 — ExecuTorch + CUDA backend (AOTInductor).

torch.export -> CudaPartitioner (whole graph to AOTInductor; Triton kernels, cuBLAS where shimmed)
-> .pte (program + AOTI .so) and .ptd (weights).  One .pte per method, each in its own
directory because the CUDA backend writes a fixed-name aoti_cuda_blob.ptd.

    python -m pi05_edge.export_et_cuda --method vision --out models/et_cuda
"""
import argparse
import time
from pathlib import Path

import torch

from .common import MODELS, mem_report, write_result
from .model import Pi05Parts, load_pi05


def io_config(mode: str):
    """Where the program boundary lives.

    device     inputs and outputs stay on the GPU (needs a runtime that can hand over device buffers:
               the C++ Module API; the 1.5.1 Python bindings allocate outputs on the host and fail with 0x12)
    device-in  inputs are taken from the GPU, outputs are copied to the host by the program
    host       inputs copied host->device and outputs device->host inside the program (classic)
    """
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    if mode == "host":
        return ExecutorchBackendConfig()
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    return ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=(mode == "device")),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=(mode != "device")),
    )


def patch_adarms_chunk():
    """PiGemmaRMSNorm.forward with `modulation[..., a:b]` instead of `modulation.chunk(3, dim=-1)` (same values)."""
    import torch
    from lerobot.policies.pi_gemma import PiGemmaRMSNorm

    def forward(self, x, cond=None):
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)
        d = self.dim
        scale, shift, gate = modulation[..., :d], modulation[..., d:2 * d], modulation[..., 2 * d:]
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    PiGemmaRMSNorm.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="vision | prefix | prefix_embed | prefix_l<i> | denoise_step | denoise10")
    ap.add_argument("--prefix-chunks", type=int, default=3)
    ap.add_argument("--out", default=str(MODELS / "et_cuda"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--io", default="device-in", choices=["device", "device-in", "host"], help="program-boundary I/O placement (see io_config)")
    ap.add_argument("--triton", default="auto", choices=["auto", "on", "off"],
                    help="ExecuTorch's Triton SDPA kernel needs bf16; the fp32 SigLIP tower must use the decomposed path (auto: off for vision)")
    ap.add_argument("--tf32", action="store_true", help="compile fp32 GEMMs with TF32 tensor cores (TensorRT's default for fp32)")
    ap.add_argument("--split-to-slices", action="store_true",
                    help="express AdaRMS's chunk(3) as three slices; ExecuTorch 1.4.1's CUDA backend has no kernel for aten::split_copy")
    args = ap.parse_args()
    if args.split_to_slices:
        patch_adarms_chunk()
    triton = args.triton if args.triton != "auto" else ("off" if args.method == "vision" else "on")
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True

    import executorch
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.backend.compile_spec_schema import CompileSpec

    et_version = getattr(executorch, "__version__", None) or __import__("importlib.metadata").metadata.version("executorch")
    out_dir = Path(args.out) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    mem_report("start")
    policy = load_pi05(dtype=args.dtype)
    parts = Pi05Parts(policy, prefix_chunks=args.prefix_chunks)
    module = parts.unit(args.method)
    example = parts.example_inputs()[args.method]
    mem_report("model loaded")

    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(module, example, strict=False)
    t_export = time.time() - t0
    n_nodes = len(ep.graph.nodes)
    print(f"[export] torch.export {args.method}: {n_nodes} nodes in {t_export:.1f}s", flush=True)
    mem_report("after torch.export")

    t0 = time.time()
    specs = [CudaBackend.generate_method_name_compile_spec("forward"),
             CompileSpec("triton_kernel_mode", triton.upper().encode())]
    partitioner = CudaPartitioner(specs)
    edge = to_edge_transform_and_lower(
        ep, partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    t_lower = time.time() - t0
    print(f"[lower] AOTInductor compile in {t_lower:.1f}s (triton_kernel_mode={triton.upper()})", flush=True)
    mem_report("after lowering")

    cfg, resident = io_config(args.io), args.io
    t0 = time.time()
    prog = edge.to_executorch(cfg)
    pte = out_dir / f"{args.method}.pte"
    with open(pte, "wb") as f:
        prog.write_to_file(f)
    prog.write_tensor_data_to_file(str(out_dir))  # external weights -> <out_dir>/aoti_cuda_blob.ptd
    t_save = time.time() - t0
    files = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(out_dir.iterdir())}
    delegates = [n.target for n in prog.exported_program().graph.nodes if "lowered_module" in str(n.target)]
    print(f"[save] {files} MB; io={resident}; lowered modules={len(delegates)}", flush=True)
    mem = mem_report("end")
    write_result(f"export_et_cuda_{et_version.split('+')[0]}_{args.method}" + ("_tf32" if args.tf32 else ""), {
        "path": "executorch_cuda", "method": args.method, "dtype": args.dtype, "executorch": et_version,
        "export_s": round(t_export, 1), "lower_s": round(t_lower, 1), "save_s": round(t_save, 1),
        "graph_nodes": n_nodes, "files_MB": files, "io": resident, "triton_kernel_mode": triton, "tf32": args.tf32, "split_to_slices": args.split_to_slices, "memory": mem,
    })


if __name__ == "__main__":
    main()
