# Stand-in with the SAM 3 interface: masks = connected bright blobs of the
# image (what a "particle" prompt would return on this test sample).
import numpy as np
from scipy import ndimage
class Sam3Processor:
    # same signature as the real one (sam3/model/sam3_image_processor.py)
    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        assert device == model["kw"].get("device"), "processor and model on different devices"
        self.model, self.confidence_threshold = model, confidence_threshold
    def set_image(self, image):
        return {"img": np.asarray(image)[..., 0].astype(float)}
    def set_text_prompt(self, prompt, state):
        img = state["img"]
        lab, n = ndimage.label(ndimage.gaussian_filter(img, 2) > 128)
        masks = np.stack([lab == k for k in range(1, n + 1)])[:, None]   # (K,1,H,W)
        scores = np.linspace(0.95, 0.4, n) if n > 1 else np.array([0.95] * n)
        keep = scores > self.confidence_threshold          # like the real processor
        return {"masks": masks[keep], "boxes": np.zeros((int(keep.sum()), 4)), "scores": scores[keep]}
