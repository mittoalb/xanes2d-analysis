"""Sample segmentation with SAM 3 (Meta, Segment Anything with Concepts).

SAM 3 runs in its own environment (Python >= 3.12, PyTorch, a CUDA GPU, the
`sam3` package and access to its Hugging Face checkpoints). xanes2d-analysis
calls that environment's Python on `_sam3_worker.py` with the image and a
text prompt, through temporary files — the same pattern as pystream's SAM 2
backend — so xanes2d-analysis itself needs no PyTorch.

The SAM 3 Python is taken from, in order: the app setting, $XANES2D_SAM3_PYTHON,
or the first `sam3` conda environment found in the usual places.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sam3_worker.py")
_CANDIDATES = ["~/miniconda3/envs/sam3/bin/python", "~/anaconda3/envs/sam3/bin/python",
               "~/conda/envs/sam3/bin/python",
               "/home/beams/USERTXM/conda/anaconda/envs/sam3/bin/python"]

SETUP_HELP = (
    "SAM 3 needs its own environment on a computer with a recent NVIDIA GPU:\n"
    "  conda create -n sam3 python=3.12 && conda activate sam3\n"
    "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
    "  git clone https://github.com/facebookresearch/sam3.git && pip install -e sam3\n"
    "  request access at https://huggingface.co/facebook/sam3, then: hf auth login\n"
    "Then set its Python in 'SAM 3 settings' (or $XANES2D_SAM3_PYTHON).")


def find_python(configured: Optional[str] = None) -> Optional[str]:
    for p in [configured, os.environ.get("XANES2D_SAM3_PYTHON")] + \
             [os.path.expanduser(c) for c in _CANDIDATES]:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def has_sam3(python: str) -> bool:
    """True if that Python can see the sam3 package (no import, just files)."""
    root = os.path.dirname(os.path.dirname(python))
    return bool(glob.glob(os.path.join(root, "lib", "python3*", "site-packages", "sam3*")))


def segment(image: np.ndarray, prompt: str = "particle", python: Optional[str] = None,
            min_score: float = 0.5, timeout_s: float = 600.0,
            checkpoint: Optional[str] = None, env: Optional[dict] = None
            ) -> Tuple[np.ndarray, int, list]:
    """Union of the SAM 3 instance masks scoring >= min_score.
    Returns (mask (H, W) bool, number of instances kept, all scores).
    Raises RuntimeError with a readable message on any problem."""
    py = find_python(python)
    if py is None:
        raise RuntimeError("no SAM 3 Python found.\n\n" + SETUP_HELP)
    with tempfile.TemporaryDirectory(prefix="xanes2d_sam3_") as td:
        req = {"image_path": os.path.join(td, "image.npy"), "prompt": prompt,
               "out_masks": os.path.join(td, "masks.npy"),
               "out_info": os.path.join(td, "info.json"), "checkpoint": checkpoint}
        np.save(req["image_path"], np.asarray(image, dtype=np.float32))
        rpath = os.path.join(td, "request.json")
        with open(rpath, "w") as fh:
            json.dump(req, fh)
        try:
            proc = subprocess.run([py, WORKER, rpath], capture_output=True, text=True,
                                  timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SAM 3 did not finish within {timeout_s:.0f} s")
        if proc.returncode != 0 or not os.path.exists(req["out_masks"]):
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
            hint = ""
            if "No module named 'sam3'" in tail:
                hint = "\n\nThat Python has no sam3 package.\n\n" + SETUP_HELP
            elif "401" in tail or "gated" in tail.lower() or "access" in tail.lower():
                hint = ("\n\nThe checkpoints need Hugging Face access: request it at "
                        "https://huggingface.co/facebook/sam3 and run 'hf auth login' "
                        "in the SAM 3 environment.")
            raise RuntimeError(f"SAM 3 failed (exit {proc.returncode}):\n{tail}{hint}")
        masks = np.load(req["out_masks"])
        with open(req["out_info"]) as fh:
            scores = json.load(fh).get("scores", [])
    h, w = np.asarray(image).shape[:2]
    keep = [i for i, s in enumerate(scores) if s >= min_score] if scores else list(range(len(masks)))
    if not keep:
        return np.zeros((h, w), bool), 0, scores
    union = np.any(masks[keep], axis=0)
    if union.shape != (h, w):                 # SAM output at another size: nearest resize
        ys = (np.arange(h) * union.shape[0] / h).astype(int)
        xs = (np.arange(w) * union.shape[1] / w).astype(int)
        union = union[np.ix_(ys, xs)]
    return union, len(keep), scores
