"""SAM 3 worker, run by the SAM 3 environment's Python (not by
xanes2d-analysis's). It only needs numpy, Pillow, torch and sam3:

    python _sam3_worker.py request.json

request.json: {"image_path": .npy (H, W) float, "prompt": str,
               "out_masks": .npy path, "out_info": .json path,
               "checkpoint": optional path, "min_score": float,
               "device": "auto" | "cuda" | "cpu"}
Writes the instance masks (K, H, W) bool and {"scores": [...]} .
"""

import json
import sys

import numpy as np


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu() if str(getattr(x, "dtype", "")).startswith("torch.bfloat") else x.detach().cpu()
    return np.asarray(x)


def _to_rgb_uint8(img):
    a = np.array(img, dtype=np.float64)
    good = np.isfinite(a)
    if not good.any():
        a = np.zeros_like(a)
    else:
        lo, hi = np.percentile(a[good], (1, 99))
        a[~good] = lo
        a = np.clip((a - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    u8 = (a * 255).astype(np.uint8)
    return np.stack([u8] * 3, axis=-1)


def main(path):
    with open(path) as fh:
        req = json.load(fh)
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    img = np.load(req["image_path"])
    device = req.get("device") or "auto"
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    kwargs = {"device": device}
    if req.get("checkpoint"):
        kwargs["checkpoint_path"] = req["checkpoint"]      # else sam3.pt from facebook/sam3
    model = build_sam3_image_model(**kwargs)
    # Sam3Processor defaults to device="cuda" whatever the model's device,
    # and drops objects below its own confidence_threshold (default 0.5):
    # pass both explicitly.
    processor = Sam3Processor(model, device=device,
                              confidence_threshold=float(req.get("min_score", 0.5)))
    state = processor.set_image(Image.fromarray(_to_rgb_uint8(img)))
    out = processor.set_text_prompt(state=state, prompt=req.get("prompt") or "particle")
    masks = _to_np(out["masks"]).astype(bool)
    scores = _to_np(out["scores"]).astype(float).ravel()
    if masks.ndim == 4:                       # (K, 1, H, W)
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    np.save(req["out_masks"], masks)
    with open(req["out_info"], "w") as fh:
        json.dump({"scores": scores.tolist(), "n": int(masks.shape[0]), "device": device}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
