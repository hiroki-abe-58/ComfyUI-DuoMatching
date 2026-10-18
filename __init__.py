"""DuoMatching for ComfyUI (unofficial integration of https://github.com/JohnZhan2023/DuoMatching)."""

import logging

try:
    from .duomatching_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[DuoMatching] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
