# Vendored upstream code

Source: https://github.com/JohnZhan2023/DuoMatching, commit `1f1d81f1f786ec121e41d852dfad8bea4eb3e252` (2026-10-05), published
under Apache-2.0. `LICENSE` and `NOTICE` in this folder are upstream's files, unchanged:

| File | SHA-256 |
| --- | --- |
| `LICENSE` (Apache License 2.0) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `NOTICE` | `6f73a570a8befbb16dcdb64b0a3324a0a9bb80b1349723cccfc202d8c1dad7bb` |

The package is vendored as `dm`. Upstream's top-level names `wan`, `utils`, `pipeline` and `demo_utils` are generic and would
collide with other ComfyUI custom nodes if registered globally, so here they are subpackages of `dm`.

Only the files that text-to-video inference needs are vendored. `scripts/vendor_sync.py` rebuilds this folder from an upstream
checkout. `tests/test_vendor.py` repeats that in CI and compares every byte. `MANIFEST.sha256` lists the SHA-256 of every file here
(20 code and config files plus `LICENSE` and `NOTICE`).

## Copied unchanged (byte-identical)

- `dm/wan/modules/`: `model.py`, `t5.py`, `tokenizers.py`, `vae.py`. Their headers say "Copyright 2024-2025 The Alibaba Wan Team
  Authors", Apache-2.0. Also `__init__.py`.
- `dm/wan/__init__.py`, `dm/utils/__init__.py`, `dm/demo_utils/__init__.py`: upstream's package markers. They are a docstring or
  empty, and import nothing.
- `dm/utils/scheduler.py`, `dm/utils/checkpoint.py`. The second is upstream's strict loader for `generator_ema` weights.
- `dm/demo_utils/memory.py`. Upstream copied it from lllyasviel/FramePack (Apache-2.0), as its header and upstream's NOTICE say.
- `dm/configs/`: `default_config.yaml`, `duomatching_inference.yaml`. These are the release configs; the pack reads the inference
  keys only.

## Patched

Each patched file starts with the line `# Changed by ComfyUI-DuoMatching; duomatching_comfy/vendor/UPSTREAM.md lists the changes.`
The exact edits are the `EDITS` table of `scripts/vendor_sync.py`. Each one must match exactly once. The test applies them to the
upstream files and compares the bytes.

- `wan/modules/attention.py` (Wan team, Apache-2.0): a new function `_sdpa_flash_attention` and two lines at the top of
  `flash_attention()`.
  - When neither flash-attn 2 nor 3 is installed, the attention is computed with PyTorch `scaled_dot_product_attention`. Inputs
    are cast to half precision, as `flash_attention()` does; the output keeps the input dtype.
  - The fallback refuses padding lengths, windows, dropout and causal attention of unequal lengths. Inference uses none of them.
  - Without flash-attn, upstream stops at `assert FLASH_ATTN_2_AVAILABLE`, and flash-attn has no official Windows build.
  - With flash-attn installed, nothing changes.
- `wan/modules/causal_model.py`: two imports become package-relative (`from .attention ...`, `from .model ...`). The test compares
  the parsed syntax trees without the import statements.
- `pipeline/causal_inference.py`: two imports become package-relative. Same syntax-tree test.
- `utils/paths.py`: upstream reads the Wan2.1 model folder from the environment variable `DUOMATCHING_WAN_ROOT` and defaults to
  `wan_models`. Here, a module variable `WAN_ROOT = "wan_models"` (the same default) replaces the variable, and the pack sets it to
  the ComfyUI model folder. `import os` is removed because it is no longer used.
- `utils/wan_wrapper.py`: imports become package-relative. `WanTextEncoder.device` returns the CPU device when CUDA is not
  available; upstream calls `torch.cuda.current_device()` unconditionally. The nodes need CUDA anyway. This edit lets the test
  suite and CI run the code on the CPU.

## Pack shims (own files, not from upstream)

- `dm/__init__.py`: package marker.
- `dm/pipeline/__init__.py`: imports `CausalInferencePipeline` only. Upstream's version also imports the training pipeline
  (`self_forcing_training.py`), which is not vendored.

## Not vendored

Everything else is not vendored:
- training: `trainer/`, `model/`, `train.py`, `pipeline/self_forcing_training.py`;
- LatentBridge (`latent_bridge/`, `scripts/latent_bridge/`);
- the Qwen-Image wrappers and the other `utils/` modules;
- the prompt lists and assets.

`inference.py` is not vendored either. Its text-to-video steps are re-implemented in `duomatching_comfy/runtime.py` and compared
with the original script (docs/VERIFICATION.md).

## Weights (not vendored; downloaded by the user)

The pack ships no weights. `tools/download_model.py` fetches them at fixed revisions and checks the SHA-256 listed in
`tools/model_files.sha256`:

- `duomatching.pt` = `model.pt` of https://huggingface.co/JohnZhan/DuoMatching, revision
  `dab4b2d3d53d51cd6c442174745576b35d2470a1`.
  - 5,676,238,378 bytes, SHA-256 `d7b9fa95afee152943e8b7c0613f57218c1add884dcaf77d7558c49a4c6d7f95`.
  - The weight repository's `LICENSE` file is the Apache License 2.0, byte-identical to the code `LICENSE` above. Its model card
    has no license field.
- `Wan2.1-T2V-1.3B/`: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B (Apache-2.0). It supplies the text encoder (umT5-XXL), the VAE,
  the tokenizer and the model configuration. The DuoMatching checkpoint replaces the diffusion weights.

## Provenance of the causal-generation code

- Upstream's `NOTICE` says the video runtime and the training code derive from Causal Forcing (thu-ml/Causal-Forcing,
  Apache-2.0), including its Self Forcing foundations.
- Self Forcing (guandeh17/Self-Forcing, Apache-2.0) says in its README that it is built on the open-source implementation of
  CausVid (tianweiy/CausVid).
- CausVid's repository is published under CC BY-NC-SA 4.0, and modules of the same names exist there (`causal_model.py`,
  `causal_inference.py`, `wan_wrapper.py`, `scheduler.py`).

This pack redistributes the upstream files under the license their authors declare (Apache-2.0). It does not settle that history;
check it yourself before commercial use.
