"""Real LIBERO observations -> the exact tensors every deployment path consumes.

Reuses the 120 LIBERO scenes rendered for the earlier vla_bench study (agent-view + wrist
camera at 256², 8-D proprio state, language instruction). The state is quantile-normalised
with the openpi pi05_libero norm stats when they are present in ~/.cache/openpi, then
discretised into the prompt exactly as LeRobot's Pi05PrepareStateTokenizerProcessorStep does.
All tensors are saved once to models/inputs/obs_bundle.pt so that every environment runs on
identical bits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .common import INPUTS

VLA_BENCH_DATA = Path("/home/logan/Projects/pi-0.7/vla_bench/data")
OPENPI_NORM = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets"
PALIGEMMA_SP = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"


def _tokenizer(max_length: int):
    """HF PaliGemma tokenizer if the gated repo is accessible, else the same sentencepiece model openpi uses."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        def encode(text: str):
            enc = tok(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
            return enc["input_ids"][0], enc["attention_mask"][0].bool()

        return encode, "hf:google/paligemma-3b-pt-224"
    except Exception as e:  # gated / offline
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(PALIGEMMA_SP))

        def encode(text: str):
            ids = [sp.bos_id()] + sp.encode(text)
            ids = ids[:max_length]
            mask = torch.zeros(max_length, dtype=torch.bool)
            mask[: len(ids)] = True
            ids = ids + [0] * (max_length - len(ids))
            return torch.tensor(ids, dtype=torch.long), mask

        return encode, f"sentencepiece:{PALIGEMMA_SP.name} ({type(e).__name__})"


def _norm_stats():
    for p in OPENPI_NORM.rglob("norm_stats.json"):
        d = json.loads(p.read_text())
        d = d.get("norm_stats", d)
        return {"q01": np.array(d["state"]["q01"]), "q99": np.array(d["state"]["q99"])}, str(p)
    return None, None


def _load_image(path: Path, size: int = 224) -> torch.Tensor:
    from PIL import Image
    from lerobot.policies.common.vla_utils import resize_with_pad_torch

    img = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).float() / 255.0  # (H,W,3) in [0,1]
    img = img.permute(2, 0, 1)[None]                                                       # (1,3,H,W)
    if img.shape[-2:] != (size, size):
        img = resize_with_pad_torch(img, size, size)
    return img * 2.0 - 1.0


def make_prompt(instruction: str, state: np.ndarray, max_state_dim: int = 32) -> str:
    """LeRobot Pi05PrepareStateTokenizerProcessorStep, verbatim semantics."""
    state = np.pad(state, (0, max_state_dim - len(state)))
    bins = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "


def build_bundle(n_obs: int = 8, n_img: int = 3, max_length: int = 200, chunk: int = 50,
                 action_dim: int = 32, seed: int = 0, init_state: int | None = None) -> dict:
    """init_state=None -> the benchmark's core initial state (evaluation set); another value -> a disjoint
    set of scenes, used as calibration data for static quantisation."""
    manifest = json.loads((VLA_BENCH_DATA / "manifest.json").read_text())
    want = manifest["core_init_state"] if init_state is None else init_state
    obs = [o for o in manifest["observations"] if o["init_state"] == want]
    # spread over the four suites
    by_suite: dict[str, list] = {}
    for o in obs:
        by_suite.setdefault(o["suite"], []).append(o)
    picked = []
    while len(picked) < n_obs:
        for s in sorted(by_suite):
            if by_suite[s] and len(picked) < n_obs:
                picked.append(by_suite[s].pop(0))

    encode, tok_name = _tokenizer(max_length)
    stats, stats_path = _norm_stats()
    g = torch.Generator().manual_seed(seed)

    images, tokens, lang_mask, prompts, uids, noises = [], [], [], [], [], []
    for o in picked:
        agent = _load_image(VLA_BENCH_DATA / o["agentview_path"])
        wrist = _load_image(VLA_BENCH_DATA / o["wrist_path"])
        empty = torch.full_like(agent, -1.0)
        imgs = torch.cat([agent, wrist] + [empty] * (n_img - 2), dim=0)      # (N,3,224,224)
        state = np.array(o["state"], dtype=np.float64)
        if stats is not None:
            state = np.clip((state - stats["q01"]) / (stats["q99"] - stats["q01"] + 1e-8) * 2 - 1, -1, 1)
        prompt = make_prompt(o["instruction"], state)
        ids, mask = encode(prompt)
        images.append(imgs); tokens.append(ids); lang_mask.append(mask); prompts.append(prompt); uids.append(o["uid"])
        noises.append(torch.randn(1, chunk, action_dim, generator=g))

    img_masks = torch.tensor([[True, True] + [False] * (n_img - 2)])           # (1,N)
    pad_mask = torch.cat([img_masks.repeat_interleave(256, dim=1), torch.stack(lang_mask)[:1]], dim=1)
    bundle = {
        "uids": uids, "prompts": prompts, "tokenizer": tok_name, "norm_stats": stats_path,
        "images": torch.stack(images),            # (B,N,3,224,224) f32 in [-1,1]
        "tokens": torch.stack(tokens),            # (B,200) int64
        "lang_mask": torch.stack(lang_mask),      # (B,200) bool
        "img_masks": img_masks,                   # (1,N) bool
        "noise": torch.cat(noises),               # (B,50,32) f32
    }
    bundle["pad_mask"] = torch.cat([img_masks.repeat_interleave(256, dim=1).expand(len(uids), -1),
                                    bundle["lang_mask"]], dim=1)  # (B,L) bool
    return bundle


def load_bundle(device: str = "cuda", path=None) -> dict:
    b = torch.load(path or INPUTS, weights_only=False)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


def obs_inputs(bundle: dict, i: int) -> dict:
    """Inputs for observation i, all with batch size 1."""
    return {
        "images": bundle["images"][i],
        "tokens": bundle["tokens"][i : i + 1],
        "lang_mask": bundle["lang_mask"][i : i + 1],
        "img_masks": bundle["img_masks"],
        "pad_mask": bundle["pad_mask"][i : i + 1],
        "noise": bundle["noise"][i : i + 1],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--calib", type=int, default=0, help="also write a calibration bundle of this many scenes from another initial state")
    args = ap.parse_args()
    INPUTS.parent.mkdir(parents=True, exist_ok=True)
    if args.calib:
        c = build_bundle(args.calib, init_state=17, seed=1)
        torch.save(c, INPUTS.parent / "calib_bundle.pt")
        print("calibration bundle:", len(c["uids"]), "scenes from init_state 17 ->", INPUTS.parent / "calib_bundle.pt")
    b = build_bundle(args.n_obs)
    torch.save(b, INPUTS)
    print("tokenizer:", b["tokenizer"]); print("norm stats:", b["norm_stats"])
    print("uids:", b["uids"])
    print("prompt[0]:", repr(b["prompts"][0]))
    print("tokens[0][:24]:", b["tokens"][0][:24].tolist(), "n_real:", int(b["lang_mask"][0].sum()))
    for k, v in b.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:10s} {tuple(v.shape)} {v.dtype}")
    print("saved", INPUTS)
