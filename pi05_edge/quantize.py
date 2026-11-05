"""FP8 / FP4 quantisation of the π₀.5 language-model linears, one recipe per stack.

Only the Gemma linear layers of the prefix (2B) and the action expert (300M) are quantised:
q/k/v/o projections and the gate/up/down MLP projections. The SigLIP tower and projector, the
token embedding / lm_head, the (Ada)RMSNorm dense layers and the action/time projections stay at
their trained precision, following the convention used in vla_bench, ModelOpt and llm-compressor.

  fp8_modelopt  NVIDIA ModelOpt W8A8 E4M3, per-tensor static scales from max calibration on 40 LIBERO
                scenes; exported as tensorrt.quantize_op Q/DQ nodes that TensorRT turns into FP8 GEMMs.
  fp8_torchao   torchao Float8DynamicActivationFloat8Weight, per-row dynamic activation scales,
                torch._scaled_mm; the eager PyTorch FP8 baseline.
  fp4_modelopt  NVIDIA ModelOpt NVFP4_DEFAULT_CFG: W4A4 E2M1 in blocks of 16 with an E4M3 block scale and a
                per-tensor fp32 global scale (calibrated); exported as tensorrt.dynamic_block_quantize_op
                nodes that TensorRT turns into FP4 GEMMs (Blackwell only).
  fp4_torchao   torchao NVFP4DynamicActivationNVFP4Weight (use_triton_kernel=False: the MSLK Triton
                quantiser is not installed), torch._scaled_mm FP4; the eager PyTorch FP4 baseline.
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


MODELOPT_CFGS = {
    "fp8": ("FP8_DEFAULT_CFG", "ModelOpt FP8_DEFAULT_CFG (W8A8 E4M3, per-tensor static, max calibration)"),
    "fp4": ("NVFP4_DEFAULT_CFG", "ModelOpt NVFP4_DEFAULT_CFG (W4A4 E2M1, block 16, E4M3 block scales + calibrated per-tensor scale)"),
}


def modelopt_quantize(fmt: str, policy, parts, calib_bundle: dict, n_calib: int | None = None) -> dict:
    """ModelOpt PTQ of the target linears (fmt in MODELOPT_CFGS), calibrated on full policy forwards."""
    import copy

    import modelopt.torch.quantization as mtq

    from .data import obs_inputs

    cfg_name, recipe = MODELOPT_CFGS[fmt]
    cfg = copy.deepcopy(getattr(mtq, cfg_name))
    # The default cfgs enable *weight_quantizer and *input_quantizer everywhere; switch them off outside the targets.
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
    return {"recipe": recipe, "n_linears": len(keep), "n_active_quantizers": len(quantizers), "n_calib_scenes": n,
            "calibration_s": round(time.time() - t0, 1)}


def fp8_modelopt(policy, parts, calib_bundle: dict, n_calib: int | None = None) -> dict:
    return modelopt_quantize("fp8", policy, parts, calib_bundle, n_calib)


def fp4_modelopt(policy, parts, calib_bundle: dict, n_calib: int | None = None) -> dict:
    return modelopt_quantize("fp4", policy, parts, calib_bundle, n_calib)


def fp4_torchao(policy) -> dict:
    from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    names = target_linears(policy.model)
    quantize_(policy.model, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False),
              filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and is_target(fqn))
    return {"recipe": "torchao NVFP4DynamicActivationNVFP4Weight (block 16, dynamic scales, no MSLK Triton kernel)",
            "n_linears": len(names)}


def export_context(quant: str):
    """torch.export needs ModelOpt's export mode so TensorQuantizer lowers to tensorrt.quantize_op."""
    if quant.endswith("_modelopt"):
        from modelopt.torch.quantization.utils import export_torch_mode

        return export_torch_mode()
    import contextlib

    return contextlib.nullcontext()
