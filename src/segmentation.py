"""Image preprocessing, thresholding and fish contour selection."""
from __future__ import annotations

import cv2
import numpy as np


def segment(frame: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray | None]:
    blur_size = int(config.get("gaussian_blur", 5))
    if blur_size < 1:
        blur_size = 1
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)

    mode = config.get("segmentation_mode", "hsv")
    if mode == "otsu":
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        threshold_type = cv2.THRESH_BINARY
        if config.get("otsu_invert", False):
            threshold_type = cv2.THRESH_BINARY_INV
        _, mask = cv2.threshold(
            gray, 0, 255, threshold_type | cv2.THRESH_OTSU
        )
    else:
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower = np.array(config["hsv_lower"], dtype=np.uint8)
        upper = np.array(config["hsv_upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if config.get("invert_mask", False):
            mask = cv2.bitwise_not(mask)

    kernel_size = int(config.get("morphology_kernel", 7))
    kernel_size = max(1, kernel_size)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel,
        iterations=int(config.get("close_iterations", 2)),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel,
        iterations=int(config.get("open_iterations", 1)),
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = float(config.get("min_contour_area_px", 5000))
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect = max(width, height) / max(1, min(width, height))
        if aspect < float(config.get("min_aspect_ratio", 1.15)):
            continue
        candidates.append(contour)

    contour = max(candidates, key=cv2.contourArea) if candidates else None
    return mask, contour

