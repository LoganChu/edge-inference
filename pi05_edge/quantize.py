"""FP8 quantisation of the π₀.5 language-model linears, one recipe per stack.

Only the Gemma linear layers of the prefix (2B) and the action expert (300M) are quantised:
q/k/v/o projections and the gate/up/down MLP projections. The SigLIP tower and projector, the
token embedding / lm_head, the (Ada)RMSNorm dense layers and the action/time projections stay at
their trained precision, following the convention used in vla_bench, ModelOpt and llm-compressor.

  fp8_modelopt  NVIDIA ModelOpt W8A8 E4M3, per-tensor static scales from max calibration on 40 LIBERO
                scenes; exported as tensorrt.quantize_op Q/DQ nodes that TensorRT turns into FP8 GEMMs.
  fp8_torchao   torchao Float8DynamicActivationFloat8Weight, per-row dynamic activation scales,
                torch._scaled_mm; the eager PyTorch FP8 baseline.
"""
from __future__ import annotations

import fnmatch
import time

import torch
from torch import nn

TARGET_PATTERNS = [
    "*language_model.layers.*.self_attn.*_proj",
    "*language_model.layers.*.mlp.*_proj",
    "*gemma_expert.model.layers.*.self_attn.*_proj",
    "*gemma_expert.model.layers.*.mlp.*_proj",
]


def is_target(fqn: str) -> bool:
    return any(fnmatch.fnmatch(fqn, p) for p in TARGET_PATTERNS)


def target_linears(root: nn.Module) -> list[str]:
    return [n for n, m in root.named_modules() if isinstance(m, nn.Linear) and is_target(n)]


def fp8_torchao(policy) -> dict:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

    names = target_linears(policy.model)
    quantize_(policy.model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
              filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and is_target(fqn))
    return {"recipe": "torchao Float8DynamicActivationFloat8Weight(PerRow)", "n_linears": len(names)}


def fp8_modelopt(policy, parts, calib_bundle: dict, n_calib: int | None = None) -> dict:
    import copy

    import modelopt.torch.quantization as mtq

    from .data import obs_inputs

    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    # FP8_DEFAULT_CFG enables *weight_quantizer and *input_quantizer everywhere; switch them off outside the targets.
    keep = set(target_linears(policy.model))
    for name, mod in policy.model.named_modules():
        if isinstance(mod, nn.Linear) and name not in keep:
            cfg["quant_cfg"].append({"quantizer_name": f"{name}.*", "enable": False})
    n = n_calib or len(calib_bundle["uids"])
    t0 = time.time()

    def forward_loop(model):
        with torch.no_grad():
            for i in range(n):
                parts.run_modular(obs_inputs(calib_bundle, i))  # vision + prefix + all 10 denoising steps

    mtq.quantize(policy.model, cfg, forward_loop)
    quantizers = [m for m in policy.model.modules() if type(m).__name__ == "TensorQuantizer" and not m._disabled]
    return {"recipe": "ModelOpt FP8_DEFAULT_CFG (W8A8 E4M3, per-tensor static, max calibration)",
            "n_linears": len(keep), "n_active_quantizers": len(quantizers), "n_calib_scenes": n,
            "calibration_s": round(time.time() - t0, 1)}


def export_context(quant: str):
    """torch.export needs ModelOpt's export mode so TensorQuantizer lowers to tensorrt.quantize_op."""
    if quant == "fp8_modelopt":
        from modelopt.torch.quantization.utils import export_torch_mode

        return export_torch_mode()
    import contextlib

    return contextlib.nullcontext()
