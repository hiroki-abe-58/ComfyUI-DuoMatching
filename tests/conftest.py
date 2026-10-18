"""Test setup. References come from environment variables, never from this repository:

DUOMATCHING_COMFYUI_DIR     ComfyUI checkout (v0.38.0 or later)
DUOMATCHING_UPSTREAM_DIR    JohnZhan2023/DuoMatching checkout at the commit in duomatching_comfy/vendor/UPSTREAM.md
DUOMATCHING_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
No trained weights are used: the model tests build a tiny random DuoMatching generator in the official file layout (tests/tiny_model.py)
and run the vendored pipeline on the CPU with small stand-ins for the text encoder and the VAE.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("DUOMATCHING_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("DUOMATCHING_COMFYUI_DIR")
UPSTREAM = _path("DUOMATCHING_UPSTREAM_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (DUOMATCHING_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (DUOMATCHING_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return Path(_need(UPSTREAM, "DuoMatching checkout (DUOMATCHING_UPSTREAM_DIR)"))


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """{"base": <dir holding Wan2.1-T2V-1.3B/>, "checkpoint": <tiny checkpoint>} (tiny random weights)."""
    from tiny_model import make_tiny_models

    base, checkpoint = make_tiny_models(tmp_path_factory.mktemp("models") / "duomatching")
    return {"base": base, "checkpoint": checkpoint}
