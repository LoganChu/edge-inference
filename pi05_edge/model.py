"""π₀.5 (LeRobot port of openpi) loaded straight onto the GPU, plus static-shape export units.

The policy is split into three graphs that a robot controller would call in sequence:

    vision   (N,3,224,224) f32           -> image tokens (N,256,2048)        SigLIP So400m + projector, fp32
    prefix   image tokens, text tokens,  -> prefix KV cache (18,1,1,L,256)x2 Gemma-2B prefill, bf16
             pad mask
    denoise  KV cache, pad mask, x_t, t  -> velocity (1,50,32)               Gemma-300M action expert, bf16
    denoise10  KV cache, pad mask, noise -> actions  (1,50,32)               the 10 Euler steps fused in one graph

Shapes are fixed at export: N cameras (3 for pi05_libero: two real + one empty), 200 text tokens,
so L = N*256 + 200 = 968 prefix positions.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

REPO = "lerobot/pi05_libero_base"
REVISION = "5fcf06fc42b6717e22ab885fd381a096574b12c7"


# --------------------------------------------------------------------------- loading
def load_pi05(repo: str = REPO, revision: str = REVISION, dtype: str = "bfloat16", device: str = "cuda"):
    """Build the policy *on the GPU* and stream the fp32 safetensors into it tensor by tensor.

    LeRobot's own from_pretrained materialises the model on the CPU and then loads the whole
    14.5 GB fp32 state dict next to it; that is ~22 GB of host RAM, more than this box has free.
    Here the host never holds more than one tensor (2.1 GB for the embedding table).
    """
    from huggingface_hub import hf_hub_download
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors import safe_open

    t0 = time.time()
    cfg_path = Path(hf_hub_download(repo, "config.json", revision=revision))
    config = PreTrainedConfig.from_pretrained(cfg_path.parent)
    config.dtype = dtype          # checkpoint says float32 (training default); bf16 is the deployment precision
    config.device = device        # the hub config says "mps"
    config.compile_model = False

    with torch.device(device):
        policy = PI05Policy(config)
    policy.eval().requires_grad_(False)

    st_path = hf_hub_download(repo, "model.safetensors", revision=revision)
    target = policy.state_dict()  # live tensors
    loaded: set[str] = set()
    unexpected: list[str] = []
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            fixed = policy._fix_pytorch_state_dict_keys({key: tensor}, policy.config)
            for k, v in fixed.items():
                k = k if k.startswith("model.") else "model." + k
                if k not in target:
                    unexpected.append(k)
                    continue
                with torch.no_grad():
                    target[k].copy_(v)  # dtype cast + host->device copy of this one tensor
                loaded.add(k)
            del tensor, fixed
    missing = sorted(k for k in target if k not in loaded)
    # AdaRMS layer norms in the expert have no .weight in the checkpoint by design (dense.* instead)
    if missing:
        raise RuntimeError(f"{len(missing)} missing keys, e.g. {missing[:5]}")
    torch.cuda.empty_cache()  # construction happened in fp32; hand the freed half back to the device
    print(f"[load] {repo} dtype={dtype} on {device} in {time.time()-t0:.1f}s "
          f"({len(loaded)} tensors, {len(unexpected)} unexpected keys skipped: {unexpected[:3]})", flush=True)
    return policy


def param_summary(policy) -> dict:
    out = {}
    m = policy.model
    groups = {
        "vision_tower+projector": [m.paligemma_with_expert.paligemma.model.vision_tower,
                                   m.paligemma_with_expert.paligemma.model.multi_modal_projector],
        "gemma_2b_prefix": [m.paligemma_with_expert.paligemma.model.language_model],
        "gemma_300m_expert": [m.paligemma_with_expert.gemma_expert.model],
        "action/time projections": [m.action_in_proj, m.action_out_proj, m.time_mlp_in, m.time_mlp_out],
    }
    for name, mods in groups.items():
        n = sum(p.numel() for mod in mods for p in mod.parameters())
        b = sum(p.numel() * p.element_size() for mod in mods for p in mod.parameters())
        out[name] = {"params_M": round(n / 1e6, 1), "bytes_GB": round(b / 1e9, 3)}
    return out


# --------------------------------------------------------------------------- export units
class Vision(nn.Module):
    """SigLIP + projector for a batch of N camera images. Output cast to the LLM dtype."""

    def __init__(self, policy):
        super().__init__()
        self.pwe = policy.model.paligemma_with_expert
        self.out_dtype = self.pwe.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pwe.embed_image(images).to(self.out_dtype)


class Prefix(nn.Module):
    """Gemma-2B prefill over [image tokens | text tokens] -> stacked K and V for every layer."""

    def __init__(self, policy):
        super().__init__()
        self.pwe = policy.model.paligemma_with_expert
        self.lm = self.pwe.paligemma.model.language_model
        self.lm.config._attn_implementation = "eager"
        self.width = self.lm.config.hidden_size

    def forward(self, img_tokens: torch.Tensor, tokens: torch.Tensor, pad_mask: torch.Tensor):
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

        lang = self.pwe.embed_language_tokens(tokens)                       # (1,200,2048)
        embs = torch.cat([img_tokens.reshape(1, -1, self.width), lang], dim=1)  # (1,L,2048)
        att = torch.zeros_like(pad_mask, dtype=torch.long)                  # fully bidirectional prefix
        att_2d = make_att_2d_masks(pad_mask, att)
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        out = self.lm(
            inputs_embeds=embs,
            attention_mask=prepare_attention_masks_4d(att_2d),
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
        )
        keys, values = [], []
        for k, v, *_ in out.past_key_values:
            keys.append(k)
            values.append(v)
        return torch.stack(keys), torch.stack(values)                      # (18,1,1,L,256) each


class PrefixEmbed(nn.Module):
    """Everything of the prefix that is not a transformer layer: token embedding, concat with the image
    tokens, position ids and the additive 4-D attention mask (exactly what PiGemmaModel.forward builds)."""

    def __init__(self, policy):
        super().__init__()
        self.pwe = policy.model.paligemma_with_expert
        self.lm = self.pwe.paligemma.model.language_model
        self.lm.config._attn_implementation = "eager"
        self.width = self.lm.config.hidden_size
        self.llm_dtype = self.lm.layers[0].self_attn.q_proj.weight.dtype

    def forward(self, img_tokens, tokens, pad_mask):
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
        from transformers.masking_utils import create_causal_mask

        lang = self.pwe.embed_language_tokens(tokens)
        embs = torch.cat([img_tokens.reshape(1, -1, self.width), lang], dim=1).to(self.llm_dtype)
        att_2d = make_att_2d_masks(pad_mask, torch.zeros_like(pad_mask, dtype=torch.long))
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        cache_position = torch.arange(embs.shape[1], device=embs.device)
        mask = create_causal_mask(config=self.lm.config, inputs_embeds=embs, attention_mask=prepare_attention_masks_4d(att_2d),
                                  cache_position=cache_position, past_key_values=None, position_ids=position_ids)
        return embs, position_ids, mask


class PrefixLayers(nn.Module):
    """Gemma-2B layers [start, end) of the prefill; emits the new hidden state and this chunk's K/V."""

    def __init__(self, policy, start: int, end: int):
        super().__init__()
        self.lm = policy.model.paligemma_with_expert.paligemma.model.language_model
        self.lm.config._attn_implementation = "eager"
        self.start, self.end = start, end

    def forward(self, hidden, position_ids, mask):
        from transformers import DynamicCache

        cache = DynamicCache()
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        position_embeddings = self.lm.rotary_emb(hidden, position_ids)
        for layer in self.lm.layers[self.start:self.end]:
            hidden = layer(hidden, attention_mask=mask, position_ids=position_ids, past_key_values=cache,
                           use_cache=True, cache_position=cache_position, position_embeddings=position_embeddings)
        keys = torch.stack([cache.layers[i].keys for i in range(self.start, self.end)])
        values = torch.stack([cache.layers[i].values for i in range(self.start, self.end)])
        return hidden, keys, values


def _cache_from_stacked(k_all: torch.Tensor, v_all: torch.Tensor):
    from transformers import DynamicCache

    return DynamicCache(tuple((k_all[i], v_all[i], None) for i in range(k_all.shape[0])))


class DenoiseStep(nn.Module):
    """One flow-matching Euler velocity evaluation of the action expert."""

    def __init__(self, policy):
        super().__init__()
        self.m = policy.model
        self.m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

    def forward(self, k_all, v_all, pad_mask, x_t, t):
        cache = _cache_from_stacked(k_all, v_all)
        return self.m.denoise_step(prefix_pad_masks=pad_mask, past_key_values=cache, x_t=x_t, timestep=t)


class Denoise10(nn.Module):
    """All num_inference_steps Euler steps in one graph (noise -> action chunk)."""

    def __init__(self, policy):
        super().__init__()
        self.m = policy.model
        self.step = DenoiseStep(policy)
        self.num_steps = policy.config.num_inference_steps

    def forward(self, k_all, v_all, pad_mask, noise):
        dt = -1.0 / self.num_steps
        x_t = noise
        for s in range(self.num_steps):
            t = torch.full((noise.shape[0],), 1.0 + s * dt, dtype=torch.float32, device=noise.device)
            x_t = x_t + dt * self.step(k_all, v_all, pad_mask, x_t, t)
        return x_t


class Pi05Parts:
    """The three/four export units plus the host-side glue that strings them together."""

    def __init__(self, policy, prefix_chunks: int = 3):
        self.policy = policy
        self.cfg = policy.config
        self.vision = Vision(policy)
        self.prefix = Prefix(policy)
        self.denoise_step = DenoiseStep(policy)
        self.denoise10 = Denoise10(policy)
        self.prefix_embed = PrefixEmbed(policy)
        depth = self.prefix.lm.config.num_hidden_layers
        self.prefix_chunks = prefix_chunks
        bounds = [round(i * depth / prefix_chunks) for i in range(prefix_chunks + 1)]
        self.prefix_layers = [PrefixLayers(policy, a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        self.n_img = len(policy.config.image_features)
        self.n_text = policy.config.tokenizer_max_length
        self.n_img_tokens = 256
        self.prefix_len = self.n_img * self.n_img_tokens + self.n_text
        self.depth = self.prefix.lm.config.num_hidden_layers
        self.head_dim = self.prefix.lm.config.head_dim
        self.llm_dtype = self.vision.out_dtype

    def example_inputs(self, device="cuda") -> dict[str, tuple]:
        """Static example inputs for every method (shapes/dtypes are what matter)."""
        L, N = self.prefix_len, self.n_img
        img = torch.zeros(N, 3, 224, 224, dtype=torch.float32, device=device)
        img_tok = torch.zeros(N, self.n_img_tokens, self.prefix.width, dtype=self.llm_dtype, device=device)
        tokens = torch.zeros(1, self.n_text, dtype=torch.long, device=device)
        pad = torch.ones(1, L, dtype=torch.bool, device=device)
        k = torch.zeros(self.depth, 1, 1, L, self.head_dim, dtype=self.llm_dtype, device=device)
        v = torch.zeros_like(k)
        x = torch.zeros(1, self.cfg.chunk_size, self.cfg.max_action_dim, dtype=torch.float32, device=device)
        t = torch.ones(1, dtype=torch.float32, device=device)
        hidden = torch.zeros(1, L, self.prefix.width, dtype=self.llm_dtype, device=device)
        pos = torch.arange(L, device=device)[None]
        mask4d = torch.zeros(1, 1, L, L, dtype=torch.float32, device=device)
        ex = {
            "vision": (img,),
            "prefix": (img_tok, tokens, pad),
            "prefix_embed": (img_tok, tokens, pad),
            "denoise_step": (k, v, pad, x, t),
            "denoise10": (k, v, pad, x),
        }
        for i in range(self.prefix_chunks):
            ex[f"prefix_l{i}"] = (hidden, pos, mask4d)
        return ex

    def unit(self, name: str) -> nn.Module:
        if name.startswith("prefix_l"):
            return self.prefix_layers[int(name[len("prefix_l"):])]
        return getattr(self, name)

    @staticmethod
    def compose_prefix(embed, layer_fns):
        """A drop-in for the single `prefix` callable built from the chunked programs."""
        def prefix(img_tok, tokens, pad_mask):
            h, pos, mask = embed(img_tok, tokens, pad_mask)
            ks, vs = [], []
            for fn in layer_fns:
                h, k, v = fn(h, pos, mask)
                ks.append(k); vs.append(v)
            return torch.cat(ks), torch.cat(vs)
        return prefix

    def prefix_chunked(self):
        return self.compose_prefix(self.prefix_embed, self.prefix_layers)

    @torch.no_grad()
    def run_modular(self, inp: dict, fused: bool = False, fns: dict | None = None) -> torch.Tensor:
        """images/tokens -> actions through the units (or through `fns`, drop-in callables)."""
        fns = fns or {"vision": self.vision, "prefix": self.prefix,
                      "denoise_step": self.denoise_step, "denoise10": self.denoise10}
        img_tok = fns["vision"](inp["images"])
        k_all, v_all = fns["prefix"](img_tok, inp["tokens"], inp["pad_mask"])
        if fused:
            return fns["denoise10"](k_all, v_all, inp["pad_mask"], inp["noise"])
        dt = -1.0 / self.cfg.num_inference_steps
        x_t = inp["noise"]
        for s in range(self.cfg.num_inference_steps):
            t = torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device=x_t.device)
            x_t = x_t + dt * fns["denoise_step"](k_all, v_all, inp["pad_mask"], x_t, t)
        return x_t

    @torch.no_grad()
    def run_reference(self, inp: dict) -> torch.Tensor:
        """LeRobot's own sample_actions on the same inputs (the ground truth for every path)."""
        images = [inp["images"][i : i + 1] for i in range(self.n_img)]
        img_masks = [inp["img_masks"][:, i] for i in range(self.n_img)]
        return self.policy.model.sample_actions(images, img_masks, inp["tokens"], inp["lang_mask"], noise=inp["noise"])
