# Verification

What was run to check this node pack, with the raw numbers. Everything below used the real released weights unless it says
"tiny model".

## 1. Setup

- Windows 11 Pro (build 26200), Intel Core Ultra 9 285K, 64 GB RAM, NVIDIA RTX 5090 32 GB (driver 610.88).
- ComfyUI v0.38.0 (commit `6b747c0428c343e1417219641db93a4fb7cb69ae`), started with `--listen 127.0.0.1 --disable-auto-launch
  --disable-api-nodes`.
- Upstream code: JohnZhan2023/DuoMatching at `1f1d81f1f786ec121e41d852dfad8bea4eb3e252`.
- Weights (SHA-256 checked, `tools/model_files.sha256`):
  - `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512624d61f7aa208f7ea8140a131f93afc9a`
  - `JohnZhan/DuoMatching` @ `dab4b2d3d53d51cd6c442174745576b35d2470a1`: `model.pt` (`d7b9fa95…`)
- Two Python environments:

  | | Reference ("ref") | ComfyUI ("comfy") |
  | --- | --- | --- |
  | Python | 3.10.20 | 3.12.13 |
  | torch | 2.8.0+cu128 (upstream pins 2.6.0, which has no RTX 5090 kernels) | 2.14.1+cu130 |
  | diffusers | 0.38.0 (upstream pin) | 0.41.0 |
  | transformers | 4.57.1 (upstream pin) | 5.20.1 |
  | tokenizers | 0.22.2 (upstream pin) | 0.23.3 |
  | numpy | 1.24.4 (upstream pin) | 2.5.2 |

  flash-attn is not installed in either (there is no official Windows build).

## 2. Pack vs the official script (ref environment)

**Official side.** `inference.py` of the pinned commit, run unchanged through a small harness that (a) replaces
`flash_attention()` in the Wan modules with PyTorch SDPA computing the same attention (bf16 inputs, no mask, output in the input
dtype; it refuses padding lengths, windows and dropout, none of which occur) and (b) records the SHA-256 of the float32 bytes of the
`video` tensor that `inference.py` hands to `write_video` (`255 * video`). Arguments: `--config_path
configs/duomatching_inference.yaml --checkpoint_path model.pt --use_ema --data_path <prompt file> --num_output_frames N --seed S
--shared_seed`, plus `--height 384 --width 640` for B3. `--shared_seed` makes the script seed every prompt with `--seed`, which is
what the node does for every video.

**Pack side.** `scripts/run_jobs.py`: the pack's runtime (the code behind the nodes), in-process, same environment, hashing the
same tensor.

| # | Size | Latent frames | Frames | Prompt, seed | Official | Pack | Generator calls | Peak VRAM (pack) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 832 x 480 | 21 | 81 | fox, 42 | `0f1a7e92536a9290` | `0f1a7e92536a9290` | 65 | 12.73 GiB |
| B2 | 832 x 480 | 21 | 81 | boat, 42 | `1d785cdcb4475d35` | `1d785cdcb4475d35` | 65 | 12.73 GiB |
| B3 | 640 x 384 | 33 | 129 | river, 7 | `3c5f95e753984779` | `3c5f95e753984779` | 101 | 11.04 GiB |

fox = "A red fox trotting through fresh snow in a pine forest, morning light, cinematic."; boat = "A sailboat gliding across a calm
turquoise bay at sunset, gentle waves."; river = "A drone flies slowly along a winding river through autumn forests, aerial
view." (B1 and B2 were one official run with both prompts). **3 of 3 identical.** Generation time in this environment (pack side):
B1 12.6 s, B2 11.0 s, B3 14.6 s. The text encoder ran with `dynamic_swap` on both sides (30.3 GB free, below the official 40 GB
threshold).

## 3. Real ComfyUI queue (comfy environment)

ComfyUI started from a checkout with a copy of this pack in `custom_nodes`, the models in an `extra_model_paths.yaml` folder, and
jobs submitted through `POST /prompt` (`scripts/queue_client.py`). To compare exact pixels, the queue workflow saved every frame
with SaveImage in addition to Save Video; the in-process run converts frames to uint8 the same way.

| Job | In-process (same env): hash, time, peak | Queue: time, peak | Frames equal |
| --- | --- | --- | --- |
| fox, 42, 832 x 480, 21 | `456883ee734b7fed`, 11.9 s, 12.76 GiB | 12.0 s, 12.73 GiB | 81 / 81, max diff 0 |
| boat, 42, 832 x 480, 21 | `dd3ceada86da7433`, 10.8 s, 12.76 GiB | 11.0 s, 12.73 GiB | 81 / 81, max diff 0 |
| river, 7, 640 x 384, 33 | `522224cea7094c6c`, 13.3 s, 11.07 GiB | 13.7 s, 11.04 GiB | 129 / 129, max diff 0 |

Both shipped workflows ran unchanged through the queue: `t2v_basic` (81 frames, 10.9 s) and `t2v_640x384_8s` (129 frames,
13.6 s), each saved as MP4. Loading took 39.4 s (text encoder `dynamic_swap`, 30.3 GB free at load).

### Why the two environments give different videos

In the comfy environment the pack gives a different video than in the ref environment for the same seed (fox: `456883ee734b7fed`
vs `0f1a7e92536a9290`). The scene, subject and composition are the same (`docs/images/samples.png` shows the comfy-environment
videos); details drift apart over the clip. Measured on the uint8 frames, ref vs comfy:

| Job | PSNR first frame | middle frame | last frame | Mean abs. difference |
| --- | --- | --- | --- | --- |
| fox | 34.5 dB | 18.0 dB | 16.9 dB | 13.3 / 255 |
| boat | 42.4 dB | 27.6 dB | 26.4 dB | 2.6 / 255 |
| river | 36.5 dB | 23.3 dB | 21.0 dB | 6.6 / 255 |

The first frame is close and the difference grows frame by frame, as expected when an autoregressive bfloat16 model conditions
every frame on the previous ones. The code path is the same (section 2 shows the pack equals the official script, and within one
environment the pack, the queue and repeated runs are identical); the difference comes from the library versions (torch kernels,
transformers' T5 numerics).

## 4. Real jobs and edge cases (comfy environment, real queue)

| Case | Result |
| --- | --- |
| 41 latent frames (161 frames, about 10 s) | success, 27.2 s, 125 generator calls, peak 18.28 GiB |
| Interrupt (`POST /interrupt` 5 s into a job) | status interrupted between two generator calls |
| Next job after the interrupt | success, model still resident (10.7 s) |
| `width` = 840 (not a multiple of 16) | the node fails before generating: "width and height must be positive multiples of 16 (Wan latent patches), got 840x480" |
| `keep_loaded` off | success; model released after the job |
| DuoMatching Unload | success, report `unloaded: false` (the previous job had already released it), 32 MiB still allocated by PyTorch |

## 5. Tests without the weights (tiny model)

- `tests/` (52 tests, CPU):
  - the vendored files against the pinned upstream commit: copied files byte for byte, patched files = upstream + the listed
    edits, syntax trees of the model and the pipeline unchanged apart from imports, `LICENSE` / `NOTICE` hashes in UPSTREAM.md;
  - the SDPA fallback against a float64 reference;
  - the package scan for environment / network / process code, and the download manifest;
  - the runtime against the official steps re-implemented in the test (1, 5 and 4 latent frames, square and 96 x 48), generator
    call counts, seeds, cache rebuild on a size change, interrupt, bad sizes, one resident model, the official strict loader
    (FSDP key names, missing EMA weights), loading under `torch.inference_mode()`;
  - the nodes as ComfyUI loads them, including ComfyUI's own `validate_prompt` on the shipped workflows.
  The tiny model keeps the official width (1536, 12 heads; the pipeline's caches are sized for it) with one layer, 64 x 64 frames,
  and small stand-ins for the text encoder and the VAE.
- `scripts/ci_queue_check.py`: a stock ComfyUI server with the pack, the tiny model on the CPU, the shipped workflow through the
  queue: same seed gives the same decoded video, another seed a different one, a bad size fails, another size runs, Unload works.
- Both also pass with CUDA hidden (`CUDA_VISIBLE_DEVICES=-1`), as on the CPU-only CI runners.
