"""Timing, memory accounting and result files shared by every path."""
import json
import os
import resource
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "raw"
INPUTS = MODELS / "inputs" / "obs_bundle.pt"


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def cgroup_peak_gb() -> float | None:
    """memory.peak of the cgroup we run in (set up by scripts/run_limited.sh)."""
    try:
        with open("/proc/self/cgroup") as f:
            path = f.read().strip().split(":", 2)[2]
        with open(f"/sys/fs/cgroup{path}/memory.peak") as f:
            return int(f.read()) / 1e9
    except (OSError, IndexError, ValueError):
        return None


def vram_gb() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "torch_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "device_used_gb": (total - free) / 1e9,
    }


def mem_report(tag: str) -> dict:
    d = {"tag": tag, "rss_gb": round(rss_gb(), 2), "peak_rss_gb": round(peak_rss_gb(), 2)}
    cg = cgroup_peak_gb()
    if cg is not None:
        d["cgroup_peak_gb"] = round(cg, 2)
    d.update({k: round(v, 2) for k, v in vram_gb().items()})
    print(f"[mem] {d}", flush=True)
    return d


class CudaTimer:
    """Wall-clock around GPU work, synchronised with CUDA events."""

    def __init__(self):
        self.samples_ms: list[float] = []

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.samples_ms.append((time.perf_counter() - self.t0) * 1e3)

    def stats(self) -> dict:
        s = sorted(self.samples_ms)
        if not s:
            return {}
        return {
            "n": len(s),
            "mean_ms": round(statistics.fmean(s), 3),
            "median_ms": round(s[len(s) // 2], 3),
            "p90_ms": round(s[int(0.9 * (len(s) - 1))], 3),
            "min_ms": round(s[0], 3),
            "max_ms": round(s[-1], 3),
        }


def bench(fn, warmup: int = 3, iters: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    t = CudaTimer()
    for _ in range(iters):
        with t:
            fn()
    return t.stats()


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("name", name)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    payload.setdefault("env", {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": os.sys.version.split()[0],
        "mem_limit": os.environ.get("PI05_EDGE_MEM_LIMIT"),
    })
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"[result] {out}", flush=True)
    return out


def action_error(a: torch.Tensor, ref: torch.Tensor, action_dim: int = 7) -> dict:
    """Fidelity of a (B, chunk, 32) action chunk against a reference, on the real action dims."""
    a = a[..., :action_dim].float().cpu()
    ref = ref[..., :action_dim].float().cpu()
    d = (a - ref).abs()
    return {
        "mae": round(d.mean().item(), 6),
        "max_abs": round(d.max().item(), 6),
        "ref_abs_mean": round(ref.abs().mean().item(), 6),
        "gripper_sign_agree": round((torch.sign(a[..., -1]) == torch.sign(ref[..., -1])).float().mean().item(), 4),
    }
