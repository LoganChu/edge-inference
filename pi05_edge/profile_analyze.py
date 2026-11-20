"""Nsight Systems / Nsight Compute reports -> results/profile/profile.json (the data behind the profile visuals).

Reads, for each path (eager bf16, direct TensorRT bf16), whatever of these exists under profiles/:

    nsys/<run>.sqlite             CUDA trace of 10 control cycles (scripts/profile.sh trace)
    nsys/<run>_gpumetrics.sqlite  the same trace plus 20 us GPU-metrics samples (scripts/profile.sh hw)
    ncu/<run>.ncu-rep             per-kernel counters for vision + prefix + one denoising step (hw)

Every GPU activity is attributed to the NVTX stage (vision / prefix / denoise_step) whose CPU range
contains its launch call, so a stage's GPU time includes the bubbles in front of its kernels.

    python -m pi05_edge.profile_analyze
"""
from __future__ import annotations

import bisect
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"
OUT = ROOT / "results" / "profile"
RUNS = {"eager": "eager_bf16", "trt": "trt_bf16"}
STAGES = ["vision", "prefix", "denoise_step"]
NCU_PY = "/opt/nvidia/nsight-compute/2026.3.1/extras/python"

CATEGORIES = ["GEMM (tensor cores)", "GEMM/GEMV (fp32 CUDA cores)", "Attention", "Normalization",
              "Elementwise", "Casts & copies", "Reductions", "Other"]
_CAT_RULES = [  # first match wins; patterns run on the demangled name, lower-cased
    ("Attention", r"fmha|flash|attention|_mha|mha_|softmax|sdpa"),
    ("GEMM/GEMV (fp32 CUDA cores)", r"simt_sgemm|sgemm|gemvx|gemv2|gemv<|internal::gemv"),
    ("GEMM (tensor cores)", r"gemm|nvjet|xmma|wmma|tensorop|s16816|s1688|cutlass|splitkreduce|matmul|cublaslt"),
    ("Normalization", r"layer_?norm|rms_?norm|groupnorm|instancenorm|normalization"),
    ("Casts & copies", r"copy_kernel|catarray|transpose|permute|nchwtonhwc|nhwctonchw|gather|memcpy|memset"),
    ("Reductions", r"reduce|scan|mean|sum_"),
    ("Elementwise", r"elementwise|functor|pointwise|gelu|silu|add|mul|div|pow|rsqrt|tanh|sigmoid|exp"),
]


def categorize(name: str) -> str:
    n = name.lower()
    if n.startswith("__myl_"):  # TensorRT Myelin fusions name the ops they contain
        if re.search(r"mha|fmha|softmax", n):
            return "Attention"
        if re.search(r"gemm|matmul", n):
            return "GEMM (tensor cores)"
        if re.search(r"mean.*(sqrt|rsqrt)|norm", n):
            return "Normalization"
        if re.search(r"cast|copy|transpose|slice|concat|reshape", n) and not re.search(r"add|mul|sub|div|exp|tanh|erf|pow", n):
            return "Casts & copies"
        return "Elementwise"
    for cat, pat in _CAT_RULES:
        if re.search(pat, n):
            return cat
    return "Other"


_GENERIC = {"gpu_kernel_impl", "gpu_kernel_impl_nocast", "elementwise_kernel", "vectorized_elementwise_kernel",
            "unrolled_elementwise_kernel", "reduce_kernel"}


def short_name(name: str, limit: int = 90) -> str:
    """A readable kernel label: the op inside PyTorch's generic launchers, TensorRT names without their hash."""
    n = re.sub(r"\s+", " ", name)
    m = re.match(r"void cutlass::Kernel2?<(\w+)>", n)
    if m:
        return m.group(1)
    m = re.search(r"internal::gemvx::kernel<int, int, (\w+)", n)
    if m:
        return f"cublas gemv ({m.group(1)})"
    m = re.match(r"void at::native::(?:<unnamed>::)?(\w+)", n)
    if m:
        outer = m.group(1)
        ops = [x for x in re.findall(r"(\w*(?:Functor\w*|_kernel_cuda|KernelImpl|_kernel_impl\w*|MeanOps|SumOps|MaxOps))", n)
               if x not in _GENERIC and x != outer]
        first = re.search(r"c10::BFloat16|\bfloat\b", n)  # the first dtype in the template = the tensor dtype
        dt = {"c10::BFloat16": "bf16", "float": "f32"}.get(first.group(0), "") if first else ""
        conv = re.search(r"converting_copy_kernel_cuda<([\w:]+), ([\w:]+)>", n)
        if conv:
            f = lambda t: {"float": "f32", "c10::BFloat16": "bf16"}.get(t, t)
            return f"{outer} · cast {f(conv.group(1))}→{f(conv.group(2))}"
        op = ops[-1] if ops else ""  # innermost: BinaryFunctor<..., MulFunctor<float>> -> MulFunctor
        op = op.replace("binary_internal::", "")
        return f"{outer} · {op}{f' ({dt})' if dt else ''}"[:limit] if op else outer
    n = re.sub(r"^void ", "", n).replace("<unnamed>::", "")
    n = re.sub(r"\(.*$", "", n)
    n = re.sub(r"<.*$", "", n)  # template arguments
    n = re.sub(r"_0x[0-9a-f]{8,}$", "", n)
    return n[:limit]


def _union(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


# ----------------------------------------------------------------------------------------------- nsys
def load_trace(db: Path) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    S = dict(c.execute("select id, value from StringIds"))
    ranges = []
    for s, e, text, text_id, tid, dom in c.execute(
            "select start, end, text, textId, globalTid, domainId from NVTX_EVENTS where eventType = 59 and end is not null"):
        ranges.append({"start": s, "end": e, "name": text if text is not None else S.get(text_id, ""), "tid": tid, "domain": dom})
    api = {corr: (s, e, tid, S.get(nid, "")) for s, e, tid, corr, nid in
           c.execute("select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")}
    acts = []
    for s, e, corr, dname, sname, gx, gy, gz, bx, by, bz, regs in c.execute(
            "select start, end, correlationId, demangledName, shortName, gridX, gridY, gridZ, blockX, blockY, blockZ, "
            "registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL"):
        name = S.get(dname, S.get(sname, "?"))
        acts.append({"kind": "kernel", "start": s, "end": e, "corr": corr, "name": name, "cat": categorize(name),
                     "grid": gx * gy * gz, "block": bx * by * bz, "regs": regs})
    kinds = dict(c.execute("select id, label from ENUM_CUDA_MEMCPY_OPER"))
    for s, e, corr, nbytes, kind in c.execute("select start, end, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        acts.append({"kind": "memcpy", "start": s, "end": e, "corr": corr, "bytes": nbytes,
                     "name": f"memcpy {kinds.get(kind, kind)}", "cat": "Casts & copies", "copy": kinds.get(kind, str(kind))})
    for s, e, corr, nbytes in c.execute("select start, end, correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMSET"):
        acts.append({"kind": "memset", "start": s, "end": e, "corr": corr, "bytes": nbytes, "name": "memset", "cat": "Casts & copies"})
    syncs = []
    stypes = dict(c.execute("select id, label from ENUM_CUPTI_SYNC_TYPE"))
    for s, e, corr, st in c.execute("select start, end, correlationId, syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        syncs.append({"start": s, "end": e, "corr": corr, "type": stypes.get(st, str(st))})
    for a in acts + syncs:
        a["api"] = api.get(a["corr"])
    metrics = None
    tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "GPU_METRICS" in tabs:
        names = dict(c.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS"))
        metrics = defaultdict(list)
        for ts, mid, v in c.execute("select timestamp, metricId, value from GPU_METRICS order by timestamp"):
            metrics[names[mid]].append((ts, v))
    return {"ranges": ranges, "acts": acts, "syncs": syncs, "metrics": metrics, "api": api}


def attribute(tr: dict) -> list[dict]:
    """Group GPU activities into cycles and stages by the NVTX range around their launch call."""
    ours = [r for r in tr["ranges"] if r["name"] in ("cycle", *STAGES)]
    cycles = sorted((r for r in ours if r["name"] == "cycle"), key=lambda r: r["start"])
    stages = sorted((r for r in ours if r["name"] in STAGES), key=lambda r: r["start"])
    starts = [r["start"] for r in stages]
    out = []
    for i, cy in enumerate(cycles):
        out.append({"idx": i, "cpu_start": cy["start"], "cpu_end": cy["end"], "tid": cy["tid"],
                    "stages": [r for r in stages if cy["start"] <= r["start"] and r["end"] <= cy["end"] and r["tid"] == cy["tid"]],
                    "acts": [], "syncs": []})
    cstarts = [cy["cpu_start"] for cy in out]

    def locate(t, tid):
        ci = bisect.bisect_right(cstarts, t) - 1
        if ci < 0 or t > out[ci]["cpu_end"] or out[ci]["tid"] != tid:
            return None, None
        si = bisect.bisect_right(starts, t) - 1
        st = stages[si] if si >= 0 and stages[si]["end"] >= t and stages[si]["tid"] == tid else None
        return out[ci], st

    for a in tr["acts"]:
        if not a["api"]:
            continue
        cy, st = locate(a["api"][0], a["api"][2])
        if cy is None:
            continue
        a["stage"] = st["name"] if st else "glue"
        a["stage_range"] = st["start"] if st else None
        cy["acts"].append(a)
    for s in tr["syncs"]:
        if s["api"]:
            cy, st = locate(s["api"][0], s["api"][2])
            if cy is not None:
                s["stage"] = st["name"] if st else "glue"
                cy["syncs"].append(s)
    for cy in out:
        cy["acts"].sort(key=lambda a: a["start"])
    return out


def cycle_breakdown(cy: dict) -> dict:
    """GPU timeline of one cycle partitioned into stage windows: busy time by category + idle bubbles."""
    acts = cy["acts"]
    g0, g1 = acts[0]["start"], max(a["end"] for a in acts)
    # stage instance windows: from the first activity of an NVTX stage instance to the first of the next one
    inst = []
    for a in acts:
        key = (a["stage"], a["stage_range"])
        if not inst or inst[-1]["key"] != key:
            inst.append({"key": key, "stage": a["stage"], "acts": []})
        inst[-1]["acts"].append(a)
    res = {s: defaultdict(float) for s in STAGES + ["glue"]}
    for j, w in enumerate(inst):
        ws = w["acts"][0]["start"]
        we = inst[j + 1]["acts"][0]["start"] if j + 1 < len(inst) else g1
        r = res[w["stage"]]
        r["window_ms"] += (we - ws) / 1e6
        busy = _union([(max(a["start"], ws), min(a["end"], we)) for a in w["acts"] if a["end"] > ws and a["start"] < we])
        r["busy_ms"] += busy / 1e6
        for a in w["acts"]:
            r[f"cat:{a['cat']}"] += (a["end"] - a["start"]) / 1e6
            r["n_kernels" if a["kind"] == "kernel" else f"n_{a['kind']}"] += 1
    # idle gaps between consecutive activities: did the GPU wait for the CPU to launch the next one?
    gaps = {s: {"host_bound_ms": 0.0, "dispatch_ms": 0.0, "n_host_bound": 0} for s in STAGES + ["glue"]}
    end_so_far = acts[0]["end"]
    for prev_end, a in zip([None] + [x["end"] for x in acts[:-1]], acts):
        if prev_end is None:
            continue
        end_so_far = max(end_so_far, prev_end)
        gap = a["start"] - end_so_far
        if gap <= 0:
            continue
        g = gaps[a["stage"]]
        if a["api"][0] > end_so_far:  # launch call issued after the GPU had already run dry
            g["host_bound_ms"] += gap / 1e6
            g["n_host_bound"] += 1
        else:
            g["dispatch_ms"] += gap / 1e6
    for s in res:
        res[s]["idle_ms"] = res[s]["window_ms"] - res[s]["busy_ms"]
        res[s].update(gaps[s])
    return {"cpu_ms": (cy["cpu_end"] - cy["cpu_start"]) / 1e6, "wall_ms": (max(g1, cy["cpu_end"]) - cy["cpu_start"]) / 1e6,
            "gpu_span_ms": (g1 - g0) / 1e6,
            "gpu_busy_ms": _union([(a["start"], a["end"]) for a in acts]) / 1e6,
            "launch_lead_ms": (g0 - cy["cpu_start"]) / 1e6, "stages": {s: dict(v) for s, v in res.items()}}


def analyze_trace(tr: dict, label: str) -> dict:
    cycles = attribute(tr)
    cycles = [c for c in cycles if c["acts"]]
    br = [cycle_breakdown(c) for c in cycles]
    n = len(br)
    med = sorted(range(n), key=lambda i: br[i]["wall_ms"])[n // 2]

    def mean(f):
        return statistics.fmean(f(b) for b in br)

    stages = {}
    for s in STAGES + ["glue"]:
        keys = set().union(*(b["stages"][s].keys() for b in br))
        stages[s] = {k: round(mean(lambda b: b["stages"][s].get(k, 0.0)), 4) for k in sorted(keys)}
    # per-kernel aggregate over all cycles
    agg = defaultdict(lambda: {"count": 0, "time_ms": 0.0, "stages": defaultdict(float)})
    hist_edges = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")]
    hist = {s: {"count": [0] * (len(hist_edges) - 1), "time_ms": [0.0] * (len(hist_edges) - 1)} for s in STAGES + ["glue"]}
    api_cost = defaultdict(lambda: {"calls": 0, "ms": 0.0})
    for c in cycles:
        for a in c["acts"]:
            d = (a["end"] - a["start"]) / 1e6
            if a["kind"] == "kernel":
                k = agg[a["name"]]
                k["count"] += 1; k["time_ms"] += d; k["stages"][a["stage"]] += d
                k["cat"] = a["cat"]; k["grid"] = a["grid"]; k["block"] = a["block"]; k["regs"] = a["regs"]
                us = d * 1e3
                b = bisect.bisect_right(hist_edges, us) - 1
                hist[a["stage"]]["count"][b] += 1
                hist[a["stage"]]["time_ms"][b] += d
            if a["api"]:
                api_cost[a["stage"]]["calls"] += 1
                api_cost[a["stage"]]["ms"] += (a["api"][1] - a["api"][0]) / 1e6
    top = sorted(agg.items(), key=lambda kv: -kv[1]["time_ms"])
    kernels = [{"name": short_name(nm), "full": nm[:300], "cat": v["cat"], "per_cycle": round(v["count"] / n, 1),
                "ms_per_cycle": round(v["time_ms"] / n, 4), "avg_us": round(v["time_ms"] / v["count"] * 1e3, 2),
                "stage": max(v["stages"], key=v["stages"].get), "grid": v["grid"], "block": v["block"], "regs": v["regs"]}
               for nm, v in top]
    syncs = defaultdict(lambda: {"count": 0, "ms": 0.0})
    for c in cycles:
        for s in c["syncs"]:
            syncs[f"{s['stage']}|{s['type']}"]["count"] += 1
            syncs[f"{s['stage']}|{s['type']}"]["ms"] += (s["end"] - s["start"]) / 1e6
    copies = defaultdict(lambda: {"count": 0, "bytes": 0, "ms": 0.0})
    for c in cycles:
        for a in c["acts"]:
            if a["kind"] == "memcpy":
                x = copies[f"{a['stage']}|{a['copy']}"]
                x["count"] += 1; x["bytes"] += a["bytes"]; x["ms"] += (a["end"] - a["start"]) / 1e6
    # timeline of the median cycle, relative to its CPU start (us)
    c = cycles[med]
    t0 = c["cpu_start"]
    cat_idx = {k: i for i, k in enumerate(CATEGORIES)}
    st_idx = {k: i for i, k in enumerate(STAGES + ["glue"])}
    inst_of, seen = {}, defaultdict(int)  # n-th instance of each stage in the cycle (denoise step 0..9)
    for r in sorted(c["stages"], key=lambda r: r["start"]):
        inst_of[r["start"]] = seen[r["name"]]
        seen[r["name"]] += 1
    timeline = {
        "fields": ["start_us", "dur_us", "category", "stage", "instance"],
        "categories": CATEGORIES, "stages": STAGES + ["glue"],
        "gpu": [[round((a["start"] - t0) / 1e3, 2), round((a["end"] - a["start"]) / 1e3, 2), cat_idx[a["cat"]], st_idx[a["stage"]],
                 inst_of.get(a["stage_range"], 0)] for a in c["acts"]],
        "cpu_stages": [[r["name"], round((r["start"] - t0) / 1e3, 1), round((r["end"] - r["start"]) / 1e3, 1)] for r in c["stages"]],
        "syncs": [[round((s["start"] - t0) / 1e3, 1), round((s["end"] - s["start"]) / 1e3, 1), s["type"]] for s in c["syncs"]],
        "cpu_ms": round((c["cpu_end"] - c["cpu_start"]) / 1e6, 3),
        "wall_ms": round(br[med]["wall_ms"], 3),
    }
    out = {
        "label": label, "n_cycles": n,
        "cycle_wall_ms": {"median": round(statistics.median(b["wall_ms"] for b in br), 3), "min": round(min(b["wall_ms"] for b in br), 3),
                          "max": round(max(b["wall_ms"] for b in br), 3)},
        "cycle_cpu_enqueue_ms": round(statistics.median(b["cpu_ms"] for b in br), 3),
        "cycle_gpu_span_ms": round(mean(lambda b: b["gpu_span_ms"]), 3),
        "cycle_gpu_busy_ms": round(mean(lambda b: b["gpu_busy_ms"]), 3),
        "launch_lead_ms": round(mean(lambda b: b["launch_lead_ms"]), 3),
        "stages": stages,
        "kernels_per_cycle": round(sum(v["count"] for v in agg.values()) / n, 1),
        "distinct_kernels": len(agg),
        "top_kernels": kernels[:40],
        "category_ms": {cat: round(sum(k["ms_per_cycle"] for k in kernels if k["cat"] == cat), 4) for cat in CATEGORIES},
        "duration_hist": {"edges_us": hist_edges[:-1], **{s: {"count": [round(x / n, 1) for x in h["count"]],
                                                              "time_ms": [round(x / n, 4) for x in h["time_ms"]]} for s, h in hist.items()}},
        "api_per_cycle": {s: {"calls": round(v["calls"] / n, 1), "ms": round(v["ms"] / n, 3)} for s, v in api_cost.items()},
        "syncs_per_cycle": {k: {"count": round(v["count"] / n, 1), "ms": round(v["ms"] / n, 3)} for k, v in syncs.items()},
        "copies_per_cycle": {k: {"count": round(v["count"] / n, 1), "MB": round(v["bytes"] / n / 1e6, 3), "ms": round(v["ms"] / n, 4)}
                             for k, v in copies.items()},
        "timeline": timeline,
    }
    trt_layers = [r for r in tr["ranges"] if r["name"] not in ("cycle", *STAGES) and r["domain"] not in (0, None)]
    if trt_layers:
        out["nvtx_layer_ranges_per_cycle"] = round(len(trt_layers) / n, 1)
    return out, cycles


def analyze_gpu_metrics(tr: dict, cycles: list[dict], bin_us: float = 100.0) -> dict:
    """Mean of each sampled GPU metric inside each stage's GPU window, and a binned series of the median cycle."""
    keep = {"SMs Active [Throughput %]": "sm_active", "SM Issue [Throughput %]": "sm_issue",
            "Tensor Active [Throughput %]": "tensor_active", "Compute Warps in Flight [Throughput %]": "warps_in_flight",
            "DRAM Read Bandwidth [Throughput %]": "dram_read", "DRAM Write Bandwidth [Throughput %]": "dram_write",
            "PCIe RX Throughput [Throughput %]": "pcie_rx", "PCIe TX Throughput [Throughput %]": "pcie_tx",
            "GPC Clock Frequency [MHz]": "gpc_mhz"}
    series = {keep[k]: v for k, v in tr["metrics"].items() if k in keep}
    ts = {k: [t for t, _ in v] for k, v in series.items()}
    per_stage = {s: defaultdict(list) for s in STAGES + ["glue", "cycle"]}
    for cy in cycles:
        acts = cy["acts"]
        g1 = max(a["end"] for a in acts)
        wins, cur = [], None
        for a in acts:
            key = (a["stage"], a["stage_range"])
            if cur is None or cur[0] != key:
                cur = [key, a["start"]]
                wins.append(cur)
        bounds = [(w[0][0], w[1], (wins[j + 1][1] if j + 1 < len(wins) else g1)) for j, w in enumerate(wins)]
        for k, v in series.items():
            for stage, ws, we in bounds + [("cycle", acts[0]["start"], g1)]:
                i0, i1 = bisect.bisect_left(ts[k], ws), bisect.bisect_right(ts[k], we)
                per_stage[stage][k].extend(x for _, x in v[i0:i1])
    stage_means = {s: {k: round(statistics.fmean(v) / (1e6 if k == "gpc_mhz" and statistics.fmean(v) > 1e5 else 1), 2)
                       for k, v in d.items() if v} for s, d in per_stage.items()}  # the clock series is recorded in Hz
    # binned series for the median cycle
    med = sorted(cycles, key=lambda c: max(c["cpu_end"], max(a["end"] for a in c["acts"])) - c["cpu_start"])[len(cycles) // 2]
    g0, g1 = med["acts"][0]["start"], max(a["end"] for a in med["acts"])
    nb = int((g1 - g0) / (bin_us * 1e3)) + 1
    binned = {}
    for k, v in series.items():
        sums, cnts = [0.0] * nb, [0] * nb
        i0, i1 = bisect.bisect_left(ts[k], g0), bisect.bisect_right(ts[k], g1)
        for t, x in v[i0:i1]:
            b = int((t - g0) / (bin_us * 1e3))
            sums[b] += x; cnts[b] += 1
        scale = 1e6 if k == "gpc_mhz" else 1  # Hz -> MHz
        binned[k] = [round(s / c / scale, 1) if c else None for s, c in zip(sums, cnts)]
    # stage boundaries of the median cycle on the same axis (ms from the first kernel)
    marks, cur = [], None
    for a in med["acts"]:
        key = (a["stage"], a["stage_range"])
        if cur != key:
            cur = key
            marks.append([a["stage"], round((a["start"] - g0) / 1e6, 3)])
    return {"stage_means": stage_means, "bin_us": bin_us, "series": binned, "stage_marks": marks,
            "cycle_ms": round((g1 - g0) / 1e6, 3)}


# ----------------------------------------------------------------------------------------------- ncu
NCU_METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",  # GB202 (GDDR7) has no dram__throughput
    "l1_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "occ_achieved_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theoretical_pct": "sm__maximum_warps_per_active_cycle_pct",
    "tensor_pipe_pct": "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "tc_bf16_pct": "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed",
    "tc_tf32_pct": "sm__ops_path_tensor_op_hmma_src_tf32_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed",
    "tc_bf16_peak": "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.peak_sustained",
    "occ_lim_regs": "launch__occupancy_limit_registers",
    "occ_lim_smem": "launch__occupancy_limit_shared_mem",
    "occ_lim_warps": "launch__occupancy_limit_warps",
    "occ_lim_blocks": "launch__occupancy_limit_blocks",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "dram_read_bytes": "dram__bytes_op_read.sum",
    "dram_write_bytes": "dram__bytes_op_write.sum",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "regs": "launch__registers_per_thread",
    "waves": "launch__waves_per_multiprocessor",
    "sm_count": "device__attribute_multiprocessor_count",
    "ffma": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "fadd": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "hfma": "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "hadd": "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "hmul": "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "tensor_bf16_ops": "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "tensor_tf32_ops": "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
    "sm_hz": "sm__cycles_elapsed.avg.per_second",
    "dram_hz": "dram__cycles_elapsed.avg.per_second",
    "ffma_peak": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "dram_bytes_peak": "dram__bytes.sum.peak_sustained",
}
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def load_ncu(path: Path) -> list[dict]:
    sys.path.insert(0, NCU_PY)
    import ncu_report

    ctx = ncu_report.load_report(str(path))
    rows = []
    for ri in range(ctx.num_ranges()):
        rg = ctx.range_by_idx(ri)
        for ai in range(rg.num_actions()):
            act = rg.action_by_idx(ai)
            row = {"name": act.name(act.NameBase_DEMANGLED)}
            for k, m in NCU_METRICS.items():
                mv = act.metric_by_name(m)
                row[k] = mv.as_double() if mv is not None else None
            stalls = {}
            for m in act.metric_names():
                if m.startswith(STALL_PREFIX) and m.endswith(STALL_SUFFIX) and "_not_issued" not in m:
                    v = act.metric_by_name(m).as_double()
                    if v:
                        stalls[m[len(STALL_PREFIX):-len(STALL_SUFFIX)]] = v
            row["stalls"] = stalls
            stage = "glue"
            st = act.nvtx_state()
            if st is not None:
                for did in st.domains():
                    dom = st.domain_by_id(did)
                    names = list(dom.push_pop_ranges())
                    for nm in reversed(names):
                        if nm in STAGES:
                            stage = nm
                            break
            row["stage"] = stage
            rules = {}  # one estimate per rule title per kernel (a rule can report several messages)
            for r in act.rule_results_as_dicts():
                sp = r.get("speedup_estimation") or {}
                val = sp.get("speedup") if isinstance(sp, dict) else None
                if not val:
                    continue
                msg = r.get("rule_message") or {}
                title = (msg.get("title") if isinstance(msg, dict) else "") or r.get("name") or r.get("rule_identifier")
                text = msg.get("message", "") if isinstance(msg, dict) else str(msg)
                text = re.sub(r"@(?:section|url):([^:@]+)(?::[^@]*)?@", r"\1", text)
                if title not in rules or val > rules[title]["speedup_pct"]:
                    rules[title] = {"name": title, "speedup_pct": val, "message": text}
            row["rules"] = list(rules.values())
            rows.append(row)
    return rows


def analyze_ncu(rows: list[dict], steps: int = 10) -> dict:
    """Per-kernel hardware metrics, weighted by how often the kernel runs per cycle (denoise kernels x10)."""
    for r in rows:
        r["cat"] = categorize(r["name"])
        r["weight"] = steps if r["stage"] == "denoise_step" else 1
        r["ms"] = (r["duration_ns"] or 0) / 1e6
        flops = 2 * (r["ffma"] or 0) + (r["fadd"] or 0) + (r["fmul"] or 0) + 2 * (r["hfma"] or 0) + (r["hadd"] or 0) + (r["hmul"] or 0)
        r["flops_simt"] = flops
        r["flops_tensor"] = (r["tensor_bf16_ops"] or 0) + (r["tensor_tf32_ops"] or 0)  # ops_path counts 2 per MAC already
        r["tensor_pct"] = max(r["tc_bf16_pct"] or 0, r["tc_tf32_pct"] or 0)  # % of the tensor-core FLOP peak
        lims = {k: r[f"occ_lim_{k}"] for k in ["regs", "smem", "warps", "blocks"] if r.get(f"occ_lim_{k}") is not None}
        r["occ_limiter"] = min(lims, key=lims.get) if lims else None
        r["dram_bytes"] = (r["dram_read_bytes"] or 0) + (r["dram_write_bytes"] or 0)
        sm, mem = r["sm_pct"] or 0, r["mem_pct"] or 0
        r["bound"] = ("compute" if sm >= mem else "memory") if max(sm, mem) >= 50 else "latency"
    by_stage = {}
    for s in STAGES + ["glue", "all"]:
        rs = [r for r in rows if s == "all" or r["stage"] == s]
        w = sum(r["ms"] * r["weight"] for r in rs)
        if not w:
            continue

        def wavg(k):
            vals = [(r[k], r["ms"] * r["weight"]) for r in rs if r.get(k) is not None]
            tot = sum(x for _, x in vals)
            return round(sum(v * x for v, x in vals) / tot, 2) if tot else None

        bound = defaultdict(float)
        for r in rs:
            bound[r["bound"]] += r["ms"] * r["weight"]
        stalls = defaultdict(float)
        for r in rs:
            for k, v in r["stalls"].items():
                stalls[k] += v * r["ms"] * r["weight"]
        stall_tot = sum(stalls.values()) or 1
        by_stage[s] = {
            "kernels": len(rs), "ms_per_cycle_at_base_clock": round(w, 3),
            **{k: wavg(k) for k in ["sm_pct", "mem_pct", "dram_pct", "l2_pct", "occ_achieved_pct", "occ_theoretical_pct",
                                    "tensor_pct", "tensor_pipe_pct", "issue_active_pct", "l2_hit_pct"]},
            "bound_share": {k: round(v / w, 4) for k, v in bound.items()},
            "stall_share": dict(sorted(((k, round(v / stall_tot, 4)) for k, v in stalls.items()), key=lambda kv: -kv[1])[:8]),
            "dram_GB_per_cycle": round(sum(r["dram_bytes"] * r["weight"] for r in rs) / 1e9, 4),
            "tflop_per_cycle": round(sum((r["flops_simt"] + r["flops_tensor"]) * r["weight"] for r in rs) / 1e12, 4),
            "tensor_flop_share": round(sum(r["flops_tensor"] * r["weight"] for r in rs) /
                                       max(1, sum((r["flops_simt"] + r["flops_tensor"]) * r["weight"] for r in rs)), 4),
        }
    # rule guidance: ncu's estimated recoverable share of each kernel, summed as ms per cycle (overlapping)
    rules = defaultdict(lambda: {"ms": 0.0, "kernels": 0, "example": "", "_best": 0.0})
    for r in rows:
        for ru in r["rules"]:
            x = rules[ru["name"]]
            gain = r["ms"] * r["weight"] * ru["speedup_pct"] / 100
            x["ms"] += gain
            x["kernels"] += 1
            if gain > x["_best"]:
                x["_best"], x["example"] = gain, f"{short_name(r['name'])}: {ru['message']}"[:400]
    peaks = next((r for r in rows if r.get("ffma_peak") and r.get("sm_hz")), None)
    roof = None
    if peaks:
        roof = {"fp32_tflops": round(2 * peaks["ffma_peak"] * peaks["sm_hz"] / 1e12, 2),
                "bf16_tensor_tflops": round(peaks["tc_bf16_peak"] * peaks["sm_hz"] / 1e12, 1) if peaks.get("tc_bf16_peak") else None,
                "dram_GBps": round(peaks["dram_bytes_peak"] * peaks["dram_hz"] / 1e9, 1) if peaks.get("dram_bytes_peak") else None,
                "sm_mhz": round(peaks["sm_hz"] / 1e6), "sm_count": peaks.get("sm_count")}
    kern = []
    for r in sorted(rows, key=lambda r: -r["ms"] * r["weight"]):
        t = r["ms"] / 1e3
        kern.append({
            "name": short_name(r["name"]), "stage": r["stage"], "cat": r["cat"], "weight": r["weight"],
            "us": round(r["ms"] * 1e3, 2), **{k: (round(r[k], 2) if r[k] is not None else None) for k in
                                               ["sm_pct", "mem_pct", "dram_pct", "occ_achieved_pct", "occ_theoretical_pct",
                                                "tensor_pct", "tensor_pipe_pct", "l2_hit_pct", "waves", "issue_active_pct"]},
            "occ_limiter": r["occ_limiter"],
            "grid": int(r["grid"] or 0), "block": int(r["block"] or 0), "regs": int(r["regs"] or 0), "bound": r["bound"],
            "ai": round((r["flops_simt"] + r["flops_tensor"]) / r["dram_bytes"], 3) if r["dram_bytes"] else None,
            "tflops": round((r["flops_simt"] + r["flops_tensor"]) / t / 1e12, 3) if t else None,
            "dram_GBps": round(r["dram_bytes"] / t / 1e9, 1) if t else None,
            "top_stall": max(r["stalls"], key=r["stalls"].get) if r["stalls"] else None,
            "rules": sorted(((ru["name"], round(ru["speedup_pct"], 1)) for ru in r["rules"]), key=lambda x: -x[1])[:3],
            "ms_cycle": round(r["ms"] * r["weight"], 4),
        })
    return {"by_stage": by_stage, "roofline_peaks_base_clock": roof, "kernels": kern,
            "rules": dict(sorted(((k, {"ms": round(v["ms"], 3), "kernels": v["kernels"], "example": v["example"]}) for k, v in rules.items()),
                                 key=lambda kv: -kv[1]["ms"]))}


NCU_ONLY_FINISHED: dict = {}  # set by --skip-ncu to analyze traces while ncu is still writing


def main():
    if "--skip-ncu" in sys.argv:
        NCU_ONLY_FINISHED.update({p: True for p in RUNS})
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for path, run in RUNS.items():
        res = {}
        db = PROFILES / "nsys" / f"{run}.sqlite"
        if db.exists():
            tr = load_trace(db)
            res["trace"], _ = analyze_trace(tr, run)
            print(f"[trace] {path}: {res['trace']['n_cycles']} cycles, median {res['trace']['cycle_wall_ms']['median']} ms, "
                  f"{res['trace']['kernels_per_cycle']} kernels/cycle, GPU busy {res['trace']['cycle_gpu_busy_ms']} ms", flush=True)
        gm = PROFILES / "nsys" / f"{run}_gpumetrics.sqlite"
        if gm.exists():
            tr = load_trace(gm)
            _, cycles = analyze_trace(tr, run)
            if tr["metrics"]:
                res["gpu_metrics"] = analyze_gpu_metrics(tr, cycles)
                print(f"[gpu-metrics] {path}: {res['gpu_metrics']['stage_means']['cycle']}", flush=True)
        # a complete report first; else the kernel-replay run that was OOM-killed partway (vision + prefix only)
        cands = [PROFILES / "ncu" / f"{run}.ncu-rep", PROFILES / "ncu" / f"{run}.ncu-repz", PROFILES / "ncu" / f"{run}_kernelreplay_partial.ncu-repz"]
        rep = next((p for p in cands if p.exists()), None)
        if rep is not None and not NCU_ONLY_FINISHED.get(path, False):
            res["ncu"] = analyze_ncu(load_ncu(rep))
            res["ncu"]["source"] = rep.name
            res["ncu"]["partial"] = "partial" in rep.name
            print(f"[ncu] {path}: {len(res['ncu']['kernels'])} kernels; all: {res['ncu']['by_stage'].get('all')}", flush=True)
        result[path] = res
    out = OUT / "profile.json"
    out.write_text(json.dumps(result, separators=(",", ":")))
    print(f"[result] {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
