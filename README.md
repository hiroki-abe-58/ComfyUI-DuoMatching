# DuoMatching for ComfyUI

Text-to-video in ComfyUI with the official **DuoMatching** model
([JohnZhan2023/DuoMatching](https://github.com/JohnZhan2023/DuoMatching), paper
[arXiv:2610.03543](https://arxiv.org/abs/2610.03543), *Joint–Marginal Distribution Matching for Few-Step Video Generation*,
[project page](https://johnzhan2023.github.io/DuoMatching/)).

DuoMatching is a streaming Wan2.1-T2V-1.3B distilled from Wan2.1-T2V-14B. It generates a video frame by frame: 4 denoising steps
for the first latent frame, then 2 steps for every later one, each frame attending to the cached earlier frames. Everything runs
**inside the ComfyUI process** and needs no flash-attn, so it runs on Windows (checked) as well. The nodes never download anything.

> **Unofficial.** This node pack is an independent integration. It is not affiliated with or endorsed by the DuoMatching
> authors, by the Wan team (Alibaba) or by Comfy Org.
>
> 日本語の概要は [README.ja.md](README.ja.md) にあります。

![Five evenly spaced frames of three generated videos: a fox in snow, a sailboat at sunset (832 x 480, 81 frames each) and a river seen from above (640 x 384, 129 frames)](docs/images/samples.png)

*Rows: "A red fox trotting through fresh snow in a pine forest, morning light, cinematic." and "A sailboat gliding across a calm
turquoise bay at sunset, gentle waves." (832 x 480, 81 frames, seed 42); "A drone flies slowly along a winding river through autumn
forests, aerial view." (640 x 384, 129 frames, seed 7). Real ComfyUI queue, no cherry-picking: these are the three verification
jobs.*

## What it does

| Node | Purpose |
| --- | --- |
| **DuoMatching Loader** | Picks a DuoMatching checkpoint from `models/duomatching`, where the text encoder lives, and a keep-loaded switch |
| **DuoMatching Text to Video** | Prompt, seed, length in latent frames, width and height → `IMAGE` frames, `fps` (16) and a JSON report |
| **DuoMatching Unload** | Frees the resident model (GPU and RAM) |

Connect the frames to ComfyUI's own **Create Video** (fps from the node) and **Save Video**. The shipped workflows do exactly that.

Supported model: **`JohnZhan/DuoMatching`** `model.pt` (the released 2-step model, EMA weights), with the
**`Wan-AI/Wan2.1-T2V-1.3B`** text encoder (umT5-XXL), VAE and model configuration. Text-to-video only (upstream's image-to-video
path is not part of this pack).

## How far it was checked

Windows 11, RTX 5090 (32 GB), ComfyUI v0.38.0 (details and raw numbers in [docs/VERIFICATION.md](docs/VERIFICATION.md)):

- **Bit-identical to the official code.** In an environment with upstream's pinned libraries (Python 3.10, diffusers 0.38.0,
  transformers 4.57.1, numpy 1.24.4; torch 2.8.0 instead of the pinned 2.6.0, which has no RTX 5090 kernels) the pack gives the
  same video, byte for byte (SHA-256 of the float frames), as the official `inference.py --use_ema --shared_seed` in 3 of 3
  cases: two prompts at 832 x 480 with 81 frames, and one at 640 x 384 with 129 frames. flash-attn has no Windows build; for the
  official run its function was replaced by PyTorch SDPA computing the same attention (the same change the pack makes; see
  [vendor/UPSTREAM.md](duomatching_comfy/vendor/UPSTREAM.md)).
- **Real ComfyUI queue:** in a ComfyUI environment with torch 2.14.1, diffusers 0.41.0 and transformers 5.20.1 the queue gives the
  same frames as the pack's in-process run, byte for byte (3 of 3 jobs). Against the official run in the other environment the
  videos show the same scenes but differ in detail: an autoregressive bfloat16 model carries small numeric differences of the
  libraries from frame to frame. This is the environment, not the code; the same environment gives the same video every time.
- **Real jobs:** 5 s, 8 s and 10 s clips, two sizes, interrupt (and the next job after it), keep-loaded off, unload, and a size
  that is not a multiple of 16 (rejected before generating), all with the real weights.
- **Clean install:** fresh ComfyUI v0.38.0, fresh Python 3.12 environment, `git clone` of this repository, `pip install -r
  requirements.txt`, the model files checked with `tools/download_model.py` (SHA-256): the queue gives the same frames as the
  development environment, and both shipped workflows run unchanged and save byte-identical MP4s.
- **CI** (GitHub Actions; Ubuntu and Windows; Python 3.12 / torch 2.14.1, and Python 3.10 / torch 2.8.0 with upstream's diffusers,
  transformers and tokenizers versions): CPU tests with a tiny random generator in the official file layout, the vendored code
  compared with the pinned upstream commit, the runtime compared with the official steps, and a real ComfyUI queue on the CPU. CI
  does not download the weights (about 23 GB).

**Not checked:** Linux or macOS with a GPU, GPUs other than the 32 GB RTX 5090 (AMD / Apple GPUs in particular), other ComfyUI versions with the real model (the CPU tests and the real-queue check also run against ComfyUI v0.41.0 in CI), runs with flash-attn installed, upstream's exact torch 2.6.0, and video quality against the paper's benchmarks (the
pack reproduces the official outputs; it does not re-evaluate the model).

## Install

1. ComfyUI with the V3 node API (**v0.38.0** was used), Python 3.10+ and an NVIDIA GPU. The peak allocation was 12.7 GiB for a 5 s clip at 832 x 480
   and 18.3 GiB for 10 s (only a 32 GB card was tried). The text encoder (11 GB) stays in RAM unless the card has 40 GB free.
2. Clone this repository into `ComfyUI/custom_nodes/` and install its requirements into ComfyUI's Python:

   ```
   git clone https://github.com/hiroki-abe-58/ComfyUI-DuoMatching custom_nodes/ComfyUI-DuoMatching
   python -m pip install -r custom_nodes/ComfyUI-DuoMatching/requirements.txt
   ```

3. Download the models (about 23 GB). The script pins the revisions this pack was checked with and verifies every file's SHA-256:

   ```
   python custom_nodes/ComfyUI-DuoMatching/tools/download_model.py --models-dir models/duomatching
   ```

   Result:

   ```
   models/duomatching/
     Wan2.1-T2V-1.3B/      config.json, diffusion_pytorch_model.safetensors, models_t5_umt5-xxl-enc-bf16.pth,
                           Wan2.1_VAE.pth, google/umt5-xxl/ (tokenizer)
     duomatching.pt        = model.pt of JohnZhan/DuoMatching
   ```

   The folder name `Wan2.1-T2V-1.3B` is fixed (the official code looks it up by name). Another location works through
   `extra_model_paths.yaml` with the key `duomatching`. Any `.pt` file in that folder appears in the Loader's list. If you
   already have the Wan2.1-T2V-1.3B files (for example from ComfyUI-SGFPlus), link or copy them into
   `models/duomatching/Wan2.1-T2V-1.3B/` before running the script: files that are already there with the right content are
   kept, not downloaded again (checked with hard links), and every file is still verified.

4. Restart ComfyUI and open `workflows/gui/t2v_basic.json` (832 x 480, 5 s) or `workflows/gui/t2v_640x384_8s.json`
   (640 x 384, 8 s).

flash-attn is not needed. If it is installed, the official flash-attn path runs instead of SDPA (not checked here).

## Using the nodes

- **Length.** `latent_frames` counts Wan latent frames: video frames = 4 x latent_frames - 3. 21 gives 81 frames, about 5 s at
  16 fps (the training length and the official default). The attention cache holds every frame, so memory and time grow with the
  length (see the table below).
- **Size.** Width and height must be multiples of 16. The model was trained at 832 x 480; the official script accepts other sizes
  and so does the node (640 x 384 was checked against the official script).
- **Seed.** The same model, prompt, seed, settings and environment give the same video frames (checked). The node seeds every video
  separately, which is what the official script does with `--seed N --shared_seed` (without `--shared_seed` it seeds once per
  run, so only the first prompt of a run matches). Different library versions give a different video for the same seed (see
  above).
- **text_encoder.** `auto` follows the official rule: with less than 40 GB of free VRAM the 11 GB umT5 encoder stays in RAM and is
  streamed to the GPU when the prompt is encoded (`dynamic_swap`); otherwise it moves to the GPU. `gpu` and `dynamic_swap` force
  one.
- **Interrupt** stops between two generator calls (65 calls for a 5 s clip: 4 + 1 for the first frame, 2 + 1 for each of the
  other 20; the extra call writes the finished frame into the cache). The model stays loaded and the next job runs normally.
- **Report.** The JSON report lists the model, seed, frames, size, generator calls, seconds and peak VRAM.

## Measured on this setup

RTX 5090, the ComfyUI environment above, real queue, model resident. Loading takes about 40 s (text encoder in RAM with
`dynamic_swap`, the `auto` choice on a 32 GB card).

| Clip | Size | Latent frames | Frames | Generator calls | Generation | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| 5 s | 832 x 480 | 21 | 81 | 65 | 10.7-12.0 s | 12.7 GiB |
| 8 s | 640 x 384 | 33 | 129 | 101 | 13.6 s | 11.0 GiB |
| 10 s | 832 x 480 | 41 | 161 | 125 | 27.2 s | 18.3 GiB |

The attention cache keeps every frame, so the peak grows with the length; the 10 s clip is past the 5 s training length (it ran,
but its quality was not evaluated).

## Notes and limits

- The nodes need a CUDA GPU (the official pipeline is CUDA-only). The test suite runs a tiny random model on the CPU.
- One DuoMatching model is resident at a time; loading another checkpoint frees the first.
- RAM: the checkpoint (5.7 GB) is memory-mapped while loading; with `dynamic_swap` the text encoder (11 GB) stays in RAM.
- Dependencies: the official model code is built on diffusers; the requirements ask for at least the versions upstream pins
  (diffusers 0.38.0, transformers 4.57.1). diffusers 0.41.0 (the current release when this was written) requires
  `huggingface-hub<2.0`, so installing the requirements into a fresh ComfyUI 0.38.0 environment moves `huggingface-hub` from 2.2.0
  to 1.33.0. ComfyUI itself ran normally with it here; check other custom nodes you use if they need 2.x.
- Related: [ComfyUI-CausalForcing](https://github.com/hiroki-abe-58/ComfyUI-CausalForcing) (same author) runs the Causal Forcing
  and Causal Forcing++ checkpoints with their own official code in a WSL2 / Linux runtime. DuoMatching's code derives from
  Causal Forcing, but its repository and weights are separate; this pack runs them in-process.

## Licenses

- This node pack's own code (everything outside `duomatching_comfy/vendor`): **MIT** ([LICENSE](LICENSE)).
- `duomatching_comfy/vendor/dm`: upstream DuoMatching code, **Apache-2.0**, copied unchanged except for five documented edits
  ([vendor/UPSTREAM.md](duomatching_comfy/vendor/UPSTREAM.md)); upstream's `LICENSE` and `NOTICE` are included unchanged. Its Wan
  modules are the Wan team's (Apache-2.0).
- **Provenance note.** Upstream says its runtime derives from Causal Forcing and Self Forcing (both Apache-2.0). Self Forcing says
  it is built on CausVid, whose repository is published under **CC BY-NC-SA 4.0**. The downstream projects publish the code under
  Apache-2.0, and this pack redistributes it under that license without settling that history. Check it yourself before
  commercial use ([NOTICE](NOTICE)).
- Models (not included; downloaded only by `tools/download_model.py`): `JohnZhan/DuoMatching` (the repository's `LICENSE` file is
  the Apache License 2.0; its model card has no license field) and `Wan-AI/Wan2.1-T2V-1.3B` (Apache-2.0 according to its model
  card).

## Development

```
python -m pip install pytest ruff
DUOMATCHING_COMFYUI_DIR=<ComfyUI checkout> DUOMATCHING_UPSTREAM_DIR=<upstream checkout at the pinned commit> python -m pytest tests
python scripts/ci_queue_check.py --comfyui <ComfyUI checkout>      # real queue on the CPU with the tiny model
python scripts/vendor_sync.py --upstream <upstream checkout> --check
```

`scripts/run_jobs.py` runs the pack's runtime in-process and records hashes comparable with the official script.

## Citation

Please cite the paper ([arXiv:2610.03543](https://arxiv.org/abs/2610.03543)) if you use DuoMatching; the upstream README has no
BibTeX entry at the pinned commit.
