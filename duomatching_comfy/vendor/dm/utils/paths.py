# Changed by ComfyUI-DuoMatching; duomatching_comfy/vendor/UPSTREAM.md lists the changes.
"""Portable locations for downloaded Wan weights."""
from pathlib import Path


WAN_ROOT = "wan_models"  # folder that holds Wan2.1-T2V-1.3B (set by the node pack)


def wan_model_path(relative_path: str) -> str:
    root = Path(WAN_ROOT).expanduser()
    return str(root / relative_path)
