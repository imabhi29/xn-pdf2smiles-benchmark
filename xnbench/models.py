"""Model wrappers.

* ``DecimerDetector``  - finds and crops chemical structures on a page image
  (DECIMER Segmentation, Mask R-CNN).
* ``MolScribeRecognizer`` - converts one structure image into SMILES with a
  confidence score (MolScribe, image-to-graph).

Heavy imports happen inside the constructors, so the rest of the package
(metrics, tests, dataset loading) works without the models installed.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class DecimerDetector:
    """Detect and crop chemical structure depictions on a page image."""

    def __init__(self, expand: bool = True):
        from decimer_segmentation import segment_chemical_structures

        self._segment = segment_chemical_structures
        self._expand = expand
        self._supports_bboxes = True

    def __call__(self, page_image: np.ndarray) -> list[tuple[np.ndarray, tuple | None]]:
        """Return [(crop, bbox)], bbox = (y0, x0, y1, x1) in pixels or None."""
        if self._supports_bboxes:
            try:
                crops, bboxes = self._segment(page_image, expand=self._expand,
                                              visualization=False, return_bboxes=True)
                return list(zip(crops, bboxes))
            except TypeError:
                # Older releases cannot return boxes; labels will then be empty.
                log.warning("decimer-segmentation version has no return_bboxes; "
                            "label linking disabled. Upgrade the package to enable it.")
                self._supports_bboxes = False
        crops = self._segment(page_image, expand=self._expand, visualization=False)
        return [(c, None) for c in crops]


class MolScribeRecognizer:
    """Image -> SMILES with MolScribe."""

    def __init__(self, hf_repo: str, checkpoint: str, device: str = "cpu"):
        import torch
        from huggingface_hub import hf_hub_download
        from molscribe import MolScribe

        ckpt_path = hf_hub_download(hf_repo, checkpoint)  # cached after first run
        self._model = MolScribe(ckpt_path, device=torch.device(device))
        log.info("MolScribe loaded on %s", device)

    def predict(self, image_path: str) -> tuple[str, float | None]:
        """Return (smiles, confidence). Empty SMILES means no prediction."""
        out = self._model.predict_image_file(image_path, return_confidence=True)
        return out.get("smiles") or "", out.get("confidence")


def pad_crop(crop: np.ndarray, pad: int) -> np.ndarray:
    """Add a white border so atoms at the crop edge are not cut off."""
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    return np.pad(crop[..., :3], ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
