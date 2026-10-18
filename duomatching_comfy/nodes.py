"""ComfyUI nodes for DuoMatching text-to-video (runs in the ComfyUI process; nothing is downloaded at run time).

The output is an IMAGE batch of the frames (16 fps): connect Create Video (fps from the node) and Save Video."""

from __future__ import annotations

import json

import torch
from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

DuoMatchingModel = io.Custom("DUOMATCHING_MODEL")


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def tick(done: int):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(min(done, total), total)

    return tick


class DuoMatchingLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DuoMatchingLoader",
            display_name="DuoMatching Loader",
            category="loaders/duomatching",
            description=("Loads DuoMatching (a streaming, frame-by-frame Wan2.1-T2V-1.3B distilled from Wan2.1-T2V-14B, 2 denoising steps per "
                         "frame) with the official Wan2.1 text encoder (umT5-XXL) and VAE from models/duomatching. One model stays resident; "
                         "loading another frees it. Needs a CUDA GPU."),
            inputs=[
                io.Combo.Input("checkpoint", options=paths.checkpoints(), tooltip="A DuoMatching checkpoint (.pt) in models/duomatching"),
                io.Combo.Input("text_encoder", options=list(runtime.TEXT_ENCODER_MODES), default="auto",
                               tooltip="Where the 11 GB text encoder lives. auto = the official rule: below 40 GB of free VRAM its weights stay "
                                       "in RAM and are streamed to the GPU when used (dynamic_swap); otherwise gpu."),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Keep the model in memory after the job. Turn off to free VRAM and RAM after every run."),
            ],
            outputs=[DuoMatchingModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, checkpoint, text_encoder, keep_loaded) -> io.NodeOutput:
        return io.NodeOutput(runtime.make_handle(paths.wan_models_dir(), paths.checkpoint_path(checkpoint), text_encoder, keep_loaded))


class DuoMatchingTextToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DuoMatchingTextToVideo",
            display_name="DuoMatching Text to Video",
            category="video/duomatching",
            description=("Generates a video frame by frame: 4 denoising steps for the first latent frame, 2 for every later one (the release "
                         "setting). 21 latent frames = 81 frames = about 5 s at 16 fps; trained at 832 x 480."),
            inputs=[
                DuoMatchingModel.Input("model"),
                io.String.Input("prompt", multiline=True, default="A red fox trotting through fresh snow in a pine forest, morning light, cinematic."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFF, control_after_generate=True,
                             tooltip="Same model, prompt, seed, settings and environment give the same video."),
                io.Int.Input("latent_frames", default=21, min=1, max=120, step=1,
                             tooltip="Video length in Wan latent frames (frames = 4 x latent - 3). 21 = 81 frames, the training length; "
                                     "the attention cache grows with the length."),
                io.Int.Input("width", default=832, min=16, max=1920, step=16, tooltip="Trained at 832 x 480; other sizes work but were not trained."),
                io.Int.Input("height", default=480, min=16, max=1920, step=16),
            ],
            outputs=[io.Image.Output(display_name="images"), io.Float.Output(display_name="fps"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, prompt, seed, latent_frames, width, height) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        loaded = runtime.acquire(model)
        info = loaded.info
        try:
            out = runtime.generate(loaded, prompt, seed, latent_frames, width, height,
                                   progress=_progress(runtime.generator_calls(loaded, latent_frames)))
        finally:
            del loaded  # drop this reference first, otherwise release() cannot free the weights
            if not model.get("keep_loaded", True):
                runtime.release()
        report = json.dumps({"model": {k: info[k] for k in ("checkpoint", "text_encoder")}, **out["report"]}, indent=1)
        return io.NodeOutput(out["images"], float(runtime.FPS), report)


class DuoMatchingUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DuoMatchingUnload",
            display_name="DuoMatching Unload",
            category="loaders/duomatching",
            description="Frees the resident DuoMatching model (GPU and RAM).",
            inputs=[io.Image.Input("after", optional=True, tooltip="Optional: connect the images so this runs after them")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        was = runtime._SLOT is not None
        runtime.release()
        text = json.dumps({"unloaded": was, "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class DuoMatchingExtension(ComfyExtension):
    async def get_node_list(self):
        return [DuoMatchingLoader, DuoMatchingTextToVideo, DuoMatchingUnload]


async def comfy_entrypoint() -> DuoMatchingExtension:
    paths.register()
    return DuoMatchingExtension()
