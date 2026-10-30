"""Geometric measurements from the selected fish contour."""
from __future__ import annotations

import cv2


def measure_contour(contour, pixels_per_cm: float | None) -> dict:
    rect = cv2.minAreaRect(contour)
    (_, _), (side_a, side_b), _ = rect
    length_px = max(side_a, side_b)
    width_px = min(side_a, side_b)
    area_px = cv2.contourArea(contour)
    box = cv2.boxPoints(rect).astype("int32")

    if pixels_per_cm is None or pixels_per_cm <= 0:
        return {
            "length_px": length_px,
            "width_px": width_px,
            "area_px": area_px,
            "length_cm": None,
            "width_cm": None,
            "area_cm2": None,
            "box": box,
        }

    return {
        "length_px": length_px,
        "width_px": width_px,
        "area_px": area_px,
        "length_cm": length_px / pixels_per_cm,
        "width_cm": width_px / pixels_per_cm,
        "area_cm2": area_px / (pixels_per_cm ** 2),
        "box": box,
    }

