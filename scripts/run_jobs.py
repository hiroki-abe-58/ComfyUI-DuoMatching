"""Generate videos through this pack's runtime (the code behind the nodes), in-process, and record hashes comparable with a run of the
official inference.py.

    python scripts/run_jobs.py --wan-models <folder with Wan2.1-T2V-1.3B> --checkpoint <duomatching .pt> --jobs jobs.json --out <dir>

jobs.json: [{"id", "prompt", "seed"[, "latent_frames" (21), "width" (832), "height" (480)]}]. Writes <out>/<id>.npy (uint8 frames) and
results.jsonl with the SHA-256 (first 16 hex digits) of the float32 bytes of `255 * video` - the tensor the official inference.py hands
to write_video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duomatching_comfy import runtime  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wan-models", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--text-encoder", default="auto", choices=runtime.TEXT_ENCODER_MODES)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = runtime.acquire(runtime.make_handle(str(args.wan_models), str(args.checkpoint), args.text_encoder, True))
    print(json.dumps(loaded.info), flush=True)
    rows = []
    for job in json.loads(args.jobs.read_text(encoding="utf-8")):
        out = runtime.generate(loaded, job["prompt"], job["seed"], job.get("latent_frames", 21), job.get("width", 832), job.get("height", 480))
        arr = out["video_255"].float().numpy()
        np.save(args.out / f"{job['id']}.npy", arr.clip(0, 255).astype(np.uint8))
        row = {**job, "shape": list(arr.shape), "video_float_sha": hashlib.sha256(arr.tobytes()).hexdigest()[:16], **out["report"]}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("id", "video_float_sha", "frames", "generator_calls", "seconds", "peak_vram_allocated_gib")}), flush=True)
    with open(args.out / "results.jsonl", "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
