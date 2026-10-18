"""The runtime against the official steps, with the tiny random generator on the CPU (the trained model is compared with the official
inference script in docs/VERIFICATION.md)."""

import inspect

import pytest
import torch

import tiny_model
from duomatching_comfy import runtime
from duomatching_comfy.vendor.dm.pipeline import causal_inference

SIZE = tiny_model.SIZE


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    tiny_model.use_tiny(causal_inference, monkeypatch)
    yield
    runtime.release()


def _loaded(tiny_models, checkpoint=None, text_encoder="auto"):
    return runtime.acquire(runtime.make_handle(str(tiny_models["base"]), str(checkpoint or tiny_models["checkpoint"]), text_encoder, True,
                                               device="cpu"))


def _generate(loaded, prompt, seed, latent_frames, width=SIZE, height=SIZE, progress=None):
    """runtime.generate at the tiny size (its defaults are the official 832 x 480)."""
    return runtime.generate(loaded, prompt, seed, latent_frames, width, height, progress=progress)


def _official(tiny_models, prompt, seed, frames, width=SIZE, height=SIZE):
    """inference.py of the official repository (T2V, --use_ema, --shared_seed), step by step, without this pack's runtime."""
    from duomatching_comfy.vendor.dm.utils import paths
    from duomatching_comfy.vendor.dm.utils.checkpoint import load_generator_checkpoint

    paths.WAN_ROOT = str(tiny_models["base"])
    device = torch.device("cpu")
    pipeline = causal_inference.CausalInferencePipeline(runtime.config(), device=device)
    load_generator_checkpoint(pipeline.generator, str(tiny_models["checkpoint"]), key="generator_ema")
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    runtime.set_seed(seed)  # --shared_seed: set_seed(args.seed) before each prompt's noise
    noise = torch.randn([1, frames, 16, height // 8, width // 8], device=device, dtype=torch.bfloat16)
    video, latents = pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True, initial_latent=None, report_timing=False)
    return 255.0 * video[0].permute(0, 2, 3, 1).cpu(), latents[0].cpu()


def test_release_config():
    cfg = runtime.config()
    assert list(cfg.denoising_step_list) == [1000, 500] and list(cfg.denoising_step_list_first_chunk) == [1000, 750, 500, 250]
    assert cfg.warp_denoising_step and cfg.num_frame_per_block == 1 and not cfg.independent_first_frame and cfg.context_noise == 0
    assert cfg.model_kwargs.timestep_shift == 5.0 and (cfg.width, cfg.height) == (832, 480)


def test_the_defaults_are_the_official_ones():
    params = inspect.signature(runtime.generate).parameters
    cfg = runtime.config()
    assert params["latent_frames"].default == 21  # inference.py --num_output_frames
    assert (params["width"].default, params["height"].default) == (cfg.width, cfg.height)


def test_handle_validation():
    with pytest.raises(ValueError):
        runtime.make_handle("w", "c", "cpu", True)


@pytest.mark.parametrize(("mode", "free", "placement"), [("auto", 39.9, "dynamic_swap"), ("auto", 40.0, "gpu"), ("auto", 80.0, "gpu"),
                                                         ("gpu", 10.0, "gpu"), ("dynamic_swap", 80.0, "dynamic_swap")])
def test_text_encoder_placement_is_the_official_rule(mode, free, placement):
    assert runtime.text_encoder_placement(mode, free) == placement


def test_cuda_is_required_for_the_nodes(monkeypatch, tiny_models):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        runtime.acquire(runtime.make_handle(str(tiny_models["base"]), str(tiny_models["checkpoint"]), "auto", True))
    assert runtime._SLOT is None


def test_the_ema_weights_are_loaded_in_bfloat16(tiny_models):
    loaded = _loaded(tiny_models)
    ema = torch.load(tiny_models["checkpoint"], map_location="cpu")["generator_ema"]
    state = loaded.pipeline.generator.state_dict()
    assert state.keys() == ema.keys()
    assert all(state[k].dtype == torch.bfloat16 and torch.equal(state[k], ema[k].to(torch.bfloat16)) for k in ema)
    assert loaded.info["text_encoder"] == "cpu" and loaded.info["checkpoint"] == tiny_models["checkpoint"].name


def test_fsdp_key_names_load_the_same_weights(tiny_models, tmp_path):
    state = torch.load(tiny_models["checkpoint"], map_location="cpu")
    wrapped = {k.replace("model.", "model._fsdp_wrapped_module.", 1): v for k, v in state["generator_ema"].items()}
    path = tmp_path / "fsdp.pt"
    torch.save({"generator_ema": wrapped}, str(path))
    plain = {k: v.clone() for k, v in _loaded(tiny_models).pipeline.generator.state_dict().items()}
    runtime.release()
    fsdp = _loaded(tiny_models, checkpoint=path)
    assert all(torch.equal(plain[k], v) for k, v in fsdp.pipeline.generator.state_dict().items())


def test_a_checkpoint_without_ema_weights_is_refused(tiny_models, tmp_path):
    path = tmp_path / "no_ema.pt"
    torch.save({"generator": torch.load(tiny_models["checkpoint"], map_location="cpu")["generator_ema"]}, str(path))
    with pytest.raises(KeyError, match="generator_ema"):
        _loaded(tiny_models, checkpoint=path)
    assert runtime._SLOT is None


@pytest.mark.parametrize(("frames", "width", "height"), [(1, SIZE, SIZE), (5, SIZE, SIZE), (4, 96, 48)])
def test_generate_equals_the_official_steps(tiny_models, frames, width, height):
    out = _generate(_loaded(tiny_models), "a fox in the snow", 3, latent_frames=frames, width=width, height=height)
    runtime.release()
    video_255, latents = _official(tiny_models, "a fox in the snow", 3, frames, width, height)
    assert torch.equal(out["video_255"], video_255) and torch.equal(out["latents"], latents)
    images = out["images"]
    assert images.shape == (frames, height, width, 3) and images.dtype == torch.float32 and 0 <= images.min() and images.max() <= 1
    report = out["report"]
    assert report["generator_calls"] == 5 + (frames - 1) * 3 and report["frames"] == frames and report["fps"] == 16
    assert (report["width"], report["height"]) == (width, height) and "peak_vram_allocated_gib" not in report


def test_generator_calls_matches_the_pipeline(tiny_models):
    """4 steps + the cache pass for the first frame, 2 steps + the cache pass for every later frame; 21 latent frames = 65 calls."""
    loaded = _loaded(tiny_models)
    assert runtime.generator_calls(loaded, 21) == 65 and runtime.generator_calls(loaded, 4) == 14
    seen = []
    _generate(loaded, "a fox", 0, latent_frames=4, progress=seen.append)
    assert seen == list(range(1, 15))


def test_seed_and_prompt_change_the_video_and_repeat_exactly(tiny_models):
    loaded = _loaded(tiny_models)
    a = _generate(loaded, "a fox", 1, latent_frames=2)["video_255"]
    assert torch.equal(a, _generate(loaded, "a fox", 1, latent_frames=2)["video_255"])
    assert not torch.equal(a, _generate(loaded, "a fox", 2, latent_frames=2)["video_255"])
    assert not torch.equal(a, _generate(loaded, "a cat", 1, latent_frames=2)["video_255"])


def test_changing_the_size_rebuilds_the_caches(tiny_models):
    loaded = _loaded(tiny_models)
    first = _generate(loaded, "a river", 7, latent_frames=3)["video_255"]
    assert loaded.pipeline.kv_cache1[0]["k"].shape[1] == 3 * tiny_model.TOKENS_PER_FRAME
    wide = _generate(loaded, "a river", 7, latent_frames=2, width=96, height=48)
    assert wide["images"].shape == (2, 48, 96, 3)
    assert torch.equal(first, _generate(loaded, "a river", 7, latent_frames=3)["video_255"])


@pytest.mark.parametrize(("width", "height"), [(72, 64), (64, 40), (0, 64), (64, -16)])
def test_sizes_must_be_multiples_of_16(tiny_models, width, height):
    with pytest.raises(ValueError, match="multiples of 16"):
        _generate(_loaded(tiny_models), "a fox", 0, latent_frames=1, width=width, height=height)


def test_bad_length_and_empty_prompt_are_rejected(tiny_models):
    with pytest.raises(ValueError, match="latent_frames"):
        _generate(_loaded(tiny_models), "a fox", 0, latent_frames=0)
    with pytest.raises(ValueError, match="prompt"):
        _generate(_loaded(tiny_models), "  ", 0, latent_frames=1)


def test_interrupt_leaves_a_working_model(tiny_models):
    loaded = _loaded(tiny_models)
    first = _generate(loaded, "a fox", 5, latent_frames=4)["video_255"]

    class Stop(Exception):
        pass

    def stop(n):
        if n == 7:
            raise Stop

    with pytest.raises(Stop):
        _generate(loaded, "a fox", 5, latent_frames=4, progress=stop)
    assert "forward" not in vars(loaded.pipeline.generator)  # the counting wrapper is gone
    assert torch.equal(first, _generate(loaded, "a fox", 5, latent_frames=4)["video_255"])


def test_one_resident_model(tiny_models, tmp_path):
    a = _loaded(tiny_models)
    assert _loaded(tiny_models) is a
    other = tmp_path / "other.pt"
    other.write_bytes(tiny_models["checkpoint"].read_bytes())
    b = _loaded(tiny_models, checkpoint=other)
    assert b is not a and runtime._SLOT is b
    runtime.release()
    assert runtime._SLOT is None


def test_loading_under_inference_mode_gives_ordinary_tensors(tiny_models):
    with torch.inference_mode():
        loaded = _loaded(tiny_models)
    assert not any(p.is_inference() for p in loaded.pipeline.parameters())
