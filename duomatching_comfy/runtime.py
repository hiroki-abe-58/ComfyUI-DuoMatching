"""Loading and text-to-video generation with DuoMatching (one resident model).

The steps are those of the official `inference.py` (T2V, `--use_ema`): the pipeline is `CausalInferencePipeline` built from the
release config, the generator gets the EMA weights through the official strict loader, everything is cast to bfloat16, the text
encoder is placed on the GPU or behind the official dynamic-swap wrapper (chosen by free VRAM, as the official script does, unless
set), then for each video: seed, noise `[1, frames, 16, height / 8, width / 8]` in bfloat16, `pipeline.inference(...)`, decoded
pixels in 0..1. The model code is the vendored upstream package (vendor/UPSTREAM.md). Nothing here downloads anything.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

VENDOR = Path(__file__).resolve().parent / "vendor" / "dm"
TEXT_ENCODER_MODES = ("auto", "gpu", "dynamic_swap")
LATENT_CHANNELS = 16
FPS = 16

_LOCK = threading.Lock()
_SLOT = None


def set_seed(seed: int) -> None:
    """utils/misc.py set_seed of the official repository."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config():
    from omegaconf import OmegaConf

    return OmegaConf.merge(OmegaConf.load(VENDOR / "configs" / "default_config.yaml"), OmegaConf.load(VENDOR / "configs" / "duomatching_inference.yaml"))


@dataclass
class Loaded:
    key: tuple
    pipeline: torch.nn.Module
    device: torch.device
    info: dict = field(default_factory=dict)


def text_encoder_placement(mode: str, free_vram_gb: float) -> str:
    """The official low-memory rule (inference.py): below 40 GB of free VRAM the text encoder's weights stay in RAM and are streamed to
    the GPU when used ("dynamic_swap"); otherwise the encoder moves to the GPU ("gpu"). mode "gpu" / "dynamic_swap" forces one."""
    if mode not in TEXT_ENCODER_MODES:
        raise ValueError(f"text_encoder must be one of {TEXT_ENCODER_MODES}")
    if mode == "auto":
        return "dynamic_swap" if free_vram_gb < 40 else "gpu"
    return mode


def make_handle(wan_models: str, checkpoint: str, text_encoder: str, keep_loaded: bool, device: str = "cuda") -> dict:
    """device is always "cuda" for the nodes; the test suite runs a tiny model on the CPU."""
    if text_encoder not in TEXT_ENCODER_MODES:
        raise ValueError(f"text_encoder must be one of {TEXT_ENCODER_MODES}")
    return {"wan_models": wan_models, "checkpoint": checkpoint, "text_encoder": text_encoder, "keep_loaded": bool(keep_loaded), "device": device}


def release() -> None:
    global _SLOT
    with _LOCK:
        _SLOT = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def acquire(handle: dict) -> Loaded:
    global _SLOT
    key = (handle["wan_models"], handle["checkpoint"], handle["text_encoder"], handle.get("device", "cuda"))
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        _SLOT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ComfyUI runs nodes under torch.inference_mode(); build the weights as ordinary tensors, as the official script does.
        with torch.inference_mode(False):
            _SLOT = _load(key)
        return _SLOT


def _load(key) -> Loaded:
    wan_models, checkpoint, text_encoder_mode, device_name = key
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DuoMatching needs a CUDA GPU (the official pipeline runs on CUDA only)")
    from .vendor.dm.demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
    from .vendor.dm.pipeline.causal_inference import CausalInferencePipeline
    from .vendor.dm.utils import paths
    from .vendor.dm.utils.checkpoint import load_generator_checkpoint

    started = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device()) if device_name == "cuda" else torch.device(device_name)
    free_gb = get_cuda_free_memory_gb(device) if device.type == "cuda" else 0.0
    paths.WAN_ROOT = str(Path(wan_models))  # the folder that holds Wan2.1-T2V-1.3B
    pipeline = CausalInferencePipeline(config(), device=device)
    load_generator_checkpoint(pipeline.generator, checkpoint, key="generator_ema")  # what the official README runs (--use_ema)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    swap = device.type == "cuda" and text_encoder_placement(text_encoder_mode, free_gb) == "dynamic_swap"
    if swap:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    info = {
        "checkpoint": Path(checkpoint).name,
        "text_encoder": "dynamic_swap" if swap else ("gpu" if device.type == "cuda" else device.type),
        "free_vram_gib_at_load": round(free_gb, 1),
        "load_seconds": round(time.perf_counter() - started, 2),
    }
    return Loaded(key=key, pipeline=pipeline, device=device, info=info)


def generator_calls(loaded: Loaded, latent_frames: int) -> int:
    """Generator evaluations of one video: per block, one per denoising step plus the clean-context KV-cache pass; the first block
    uses the first-chunk schedule."""
    p = loaded.pipeline
    blocks = latent_frames // int(p.num_frame_per_block)
    first = p.denoising_step_list_first_chunk if p.denoising_step_list_first_chunk is not None else p.denoising_step_list
    return (len(first) + 1) + (blocks - 1) * (len(p.denoising_step_list) + 1)


@torch.no_grad()
def generate(loaded: Loaded, prompt: str, seed: int, latent_frames: int = 21, width: int = 832, height: int = 480, progress=None) -> dict:
    """Returns {"images": [T, height, width, 3] float in 0..1 on the CPU, "video_255": the official `255 * video` tensor (for comparisons),
    "latents", "report"}. latent_frames: Wan latent frames (video frames = 4 x latent - 3). progress(n) is called before every generator
    evaluation (it may raise to interrupt)."""
    if not prompt.strip():
        raise ValueError("prompt is empty")
    if latent_frames < 1:
        raise ValueError("latent_frames must be at least 1")
    if width % 16 or height % 16 or width < 16 or height < 16:
        raise ValueError(f"width and height must be positive multiples of 16 (Wan latent patches), got {width}x{height}")
    device = loaded.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    generator = loaded.pipeline.generator
    generator_forward = generator.forward
    calls = {"n": 0}

    def counted_forward(*args, **kwargs):
        calls["n"] += 1
        if progress is not None:
            progress(calls["n"])
        return generator_forward(*args, **kwargs)

    generator.forward = counted_forward  # the pipeline calls self.generator(...); an instance attribute is seen by Module.__call__
    try:
        set_seed(int(seed))
        noise = torch.randn([1, latent_frames, LATENT_CHANNELS, height // 8, width // 8], device=device, dtype=torch.bfloat16)
        video, latents = loaded.pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True, initial_latent=None)
    finally:
        del generator.forward  # back to the class method
        loaded.pipeline.vae.model.clear_cache()
    video = video[0].permute(0, 2, 3, 1).cpu()  # [T, H, W, C], as inference.py's rearrange 'b t c h w -> b t h w c'
    report = {
        "seed": int(seed),
        "latent_frames": int(latent_frames),
        "frames": int(video.shape[0]),
        "fps": FPS,
        "width": int(video.shape[2]),
        "height": int(video.shape[1]),
        "generator_calls": calls["n"],
        "seconds": round(time.perf_counter() - started, 2),
    }
    if device.type == "cuda":
        report["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
    return {"images": video.float(), "video_255": 255.0 * video, "latents": latents[0].cpu(), "report": report}
