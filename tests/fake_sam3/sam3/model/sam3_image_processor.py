# Stand-in with the SAM 3 interface: masks = connected bright blobs of the
# image (what a "particle" prompt would return on this test sample).
import numpy as np
from scipy import ndimage
class Sam3Processor:
    def __init__(self, model): self.model = model
    def set_image(self, image):
        return {"img": np.asarray(image)[..., 0].astype(float)}
    def set_text_prompt(self, state, prompt):
        img = state["img"]
        lab, n = ndimage.label(ndimage.gaussian_filter(img, 2) > 128)
        masks = np.stack([lab == k for k in range(1, n + 1)])[:, None]   # (K,1,H,W)
        scores = np.array([0.9] * n + [])[:n]
        return {"masks": masks, "boxes": np.zeros((n, 4)), "scores": scores}
