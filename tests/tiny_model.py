"""A tiny DuoMatching setup in the official file layout, so the real pipeline (CausalInferencePipeline, the causal Wan generator, the
KV caches, the scheduler, the official strict checkpoint loader) runs on the CPU in seconds.

- The generator keeps the official width (1536, 12 heads: the pipeline allocates its caches for that layout) but has one layer and a
  tiny FFN; it is written with `save_pretrained` into `<base>/Wan2.1-T2V-1.3B/` and its EMA weights into a checkpoint `.pt`.
- The 11 GB umT5 text encoder and the Wan VAE are replaced by small deterministic stand-ins (`StubTextEncoder`, `StubVAE`).
- Videos are made at 64 x 64 pixels (an 8 x 8 latent, 16 tokens per frame); the pipeline sizes its caches from the noise, as it does
  for any resolution. The stand-in VAE gives one pixel frame per latent frame (the real one gives 4 x latent - 3).
`use_tiny(causal_inference, patcher)` applies the stand-ins to the given module object (the one the nodes use); `patcher` is pytest's
monkeypatch or anything with a `setattr(obj, name, value)` method.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

SIZE = 64  # width and height in pixels
LATENT = (16, SIZE // 8, SIZE // 8)
TOKENS_PER_FRAME = (LATENT[1] // 2) * (LATENT[2] // 2)
WAN_CONFIG = {"dim": 1536, "eps": 1e-06, "ffn_dim": 64, "freq_dim": 256, "in_dim": 16, "model_type": "t2v", "num_heads": 12,
              "num_layers": 1, "out_dim": 16, "text_len": 512}
PLACEHOLDERS = ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl/tokenizer_config.json")


class StubTextEncoder(torch.nn.Module):
    """Deterministic [1, 512, 4096] prompt embedding derived from the prompt text."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.05))

    @property
    def device(self):
        return self.scale.device

    def forward(self, text_prompts):
        seed = int.from_bytes(hashlib.sha256("\n".join(text_prompts).encode()).digest()[:4], "little")
        g = torch.Generator().manual_seed(seed)
        emb = torch.randn(len(text_prompts), 512, 4096, generator=g).to(self.device, self.scale.dtype) * self.scale
        return {"prompt_embeds": emb}


class _StubVAEModel(torch.nn.Module):
    def clear_cache(self):
        pass


class StubVAE(torch.nn.Module):
    """decode_to_pixel: the first 3 latent channels squashed to [-1, 1] and enlarged 8 times (nearest), one frame per latent frame."""

    def __init__(self):
        super().__init__()
        self.model = _StubVAEModel()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def decode_to_pixel(self, latent, use_cache=False):
        pixels = torch.tanh(latent[:, :, :3] * self.gain).float()  # the real wrapper returns float32 too
        return pixels.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)


def make_tiny_models(base: Path, seed: int = 0) -> tuple[Path, Path]:
    """Writes <base>/Wan2.1-T2V-1.3B/ (tiny generator + placeholder files) and <base>/duomatching_tiny.pt; returns both."""
    from duomatching_comfy import runtime
    from duomatching_comfy.vendor.dm.utils import paths, wan_wrapper
    from duomatching_comfy.vendor.dm.wan.modules.causal_model import CausalWanModel

    torch.manual_seed(seed)
    wan = Path(base) / "Wan2.1-T2V-1.3B"
    wan.mkdir(parents=True, exist_ok=True)
    CausalWanModel(**WAN_CONFIG).save_pretrained(str(wan))
    for name in PLACEHOLDERS:
        (wan / name).parent.mkdir(parents=True, exist_ok=True)
        (wan / name).write_bytes(b"")
    previous = paths.WAN_ROOT
    paths.WAN_ROOT = str(base)
    try:
        generator = wan_wrapper.WanDiffusionWrapper(**runtime.config().model_kwargs, is_causal=True)
    finally:
        paths.WAN_ROOT = previous
    with torch.no_grad():
        for p in generator.parameters():
            p.copy_(torch.randn_like(p) * 0.02)
    state = {k: v.clone() for k, v in generator.state_dict().items()}
    checkpoint = Path(base) / "duomatching_tiny.pt"
    torch.save({"generator_ema": state, "generator": {k: v * 0 for k, v in state.items()}}, str(checkpoint))
    return wan.parent, checkpoint


def use_tiny(causal_inference, patcher) -> None:
    """Stand-ins for the text encoder and the VAE."""
    patcher.setattr(causal_inference, "WanTextEncoder", StubTextEncoder)
    patcher.setattr(causal_inference, "WanVAEWrapper", StubVAE)
