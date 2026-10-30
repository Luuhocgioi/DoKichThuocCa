"""Pixel-to-centimeter scale from ArUco marker or manual calibration."""
from __future__ import annotations

import cv2


class ReferenceCalibrator:
    def __init__(self, config: dict):
        self.config = config
        self.reference_type = config.get("reference_type", "aruco")
        self._aruco = None
        self._dictionary = None
        self._parameters = None

        if self.reference_type == "aruco" and hasattr(cv2, "aruco"):
            aruco = cv2.aruco
            dictionary_id = getattr(
                aruco,
                config.get("aruco_dictionary", "DICT_4X4_50"),
                aruco.DICT_4X4_50,
            )
            self._aruco = aruco
            self._dictionary = aruco.getPredefinedDictionary(dictionary_id)
            if hasattr(aruco, "DetectorParameters"):
                self._parameters = aruco.DetectorParameters()
            else:
                self._parameters = aruco.DetectorParameters_create()

    def get_scale(self, frame):
        if self.reference_type == "manual":
            scale = float(self.config.get("pixels_per_cm", 0))
            if scale > 0:
                return scale, "manual"
            return None, "manual: chua cau hinh"

        if self._aruco is None:
            return None, "ArUco khong san sang"

        corners, ids, _ = self._aruco.detectMarkers(
            frame, self._dictionary, parameters=self._parameters
        )
        if ids is None:
            return None, "khong thay marker"

        target_id = int(self.config.get("aruco_id", 0))
        physical_side_cm = float(self.config.get("aruco_side_cm", 5.0))
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) != target_id:
                continue
            points = marker_corners.reshape(4, 2)
            side_lengths = [
                ((points[i][0] - points[(i + 1) % 4][0]) ** 2
                 + (points[i][1] - points[(i + 1) % 4][1]) ** 2) ** 0.5
                for i in range(4)
            ]
            average_side_px = sum(side_lengths) / 4.0
            if physical_side_cm <= 0:
                return None, "kich thuoc marker khong hop le"
            return average_side_px / physical_side_cm, f"ArUco ID {target_id}"

        return None, f"khong thay ID {target_id}"

