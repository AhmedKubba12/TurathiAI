"""Mask post-processing (Section 3.1.3).

After the best candidate mask is selected it is refined so that the stored mask
is a clean, single, solid building region:

1.  **Morphological closing** with a large 15x15 kernel to fill small holes and
    smooth boundaries.
2.  **Largest-contour extraction** — smaller contours usually correspond to
    noise (trees, poles, background buildings) and are discarded.
3.  The largest contour is drawn as a filled polygon.
4.  The mask is **upscaled back** to the native image resolution (masks are
    generated at 1024x1024 for efficiency).
"""
from __future__ import annotations

import cv2
import numpy as np

CLOSING_KERNEL_SIZE = 15


def close_holes(mask: np.ndarray, kernel_size: int = CLOSING_KERNEL_SIZE) -> np.ndarray:
    """Morphological closing to fill small gaps and smooth edges."""
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def keep_largest_contour(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected contour, drawn as a filled polygon."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binary)
    if not contours:
        return out * 255
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(out, [largest], -1, color=1, thickness=cv2.FILLED)
    return (out * 255).astype(np.uint8)


def upscale_to(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a mask to ``(width, height)`` using nearest-neighbour."""
    width, height = size
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def refine_mask(mask: np.ndarray, native_size: tuple[int, int] | None = None,
                kernel_size: int = CLOSING_KERNEL_SIZE) -> np.ndarray:
    """Full refinement: close -> largest contour -> (optional) upscale.

    Parameters
    ----------
    mask : np.ndarray
        Binary candidate mask at working resolution.
    native_size : tuple[int, int], optional
        ``(width, height)`` of the original image; if given the refined mask is
        upscaled back to this resolution.
    kernel_size : int
        Size of the square morphological-closing kernel.

    Returns
    -------
    np.ndarray
        A single-channel uint8 mask with values in {0, 255}.
    """
    closed = close_holes(mask, kernel_size=kernel_size)
    single = keep_largest_contour(closed)
    if native_size is not None:
        single = upscale_to(single, native_size)
    return single
