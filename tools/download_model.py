"""Download the official DuoMatching checkpoint and the Wan2.1-T2V-1.3B files it needs into ComfyUI's models/duomatching, and verify
them.

    python tools/download_model.py --models-dir <ComfyUI>/models/duomatching [--verify-only]

Sources (pinned to the revisions this node pack was verified with; both Apache-2.0):
  https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B   about 17.6 GB: base model files, umT5-XXL text encoder, VAE, tokenizer
  https://huggingface.co/JohnZhan/DuoMatching     about 5.7 GB: model.pt
Result:
  models/duomatching/Wan2.1-T2V-1.3B/...   (the folder name is fixed: the official code looks it up by name)
  models/duomatching/duomatching.pt        (= model.pt)
Every file is checked against its SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with ComfyUI's
transformers). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

WAN = ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a")
DUOMATCHING = ("JohnZhan/DuoMatching", "dab4b2d3d53d51cd6c442174745576b35d2470a1")
WAN_FOLDER = "Wan2.1-T2V-1.3B"
CHECKPOINT = "duomatching.pt"
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{path below models/duomatching: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/duomatching folder")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    expected = expected_files()
    models = args.models_dir
    if not args.verify_only:
        from huggingface_hub import hf_hub_download, snapshot_download

        wan_files = sorted(name.split("/", 1)[1] for name in expected if name.startswith(WAN_FOLDER + "/"))
        snapshot_download(repo_id=WAN[0], revision=WAN[1], local_dir=str(models / WAN_FOLDER), allow_patterns=wan_files)
        target = models / CHECKPOINT
        if not (target.is_file() and sha256(target) == expected[CHECKPOINT]):
            staged = hf_hub_download(repo_id=DUOMATCHING[0], filename="model.pt", revision=DUOMATCHING[1],
                                     local_dir=str(models / ".duomatching_download"))
            os.replace(staged, target)
    ok = True
    for name, digest in expected.items():
        path = models / name
        good = path.is_file() and sha256(path) == digest
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if path.is_file() else 'MISSING ')} {name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
