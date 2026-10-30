"""Kiểm tra hình học đã biết và các trường hợp đầu vào không hợp lệ."""

import unittest

import cv2
import numpy as np

from src.main import calibrate_scale, find_fish_contour, measure_fish, preprocess, segment_fish


class MeasurementTests(unittest.TestCase):
    def test_rotated_rectangle_units(self):
        contour = cv2.boxPoints(((300, 200), (200, 60), 27)).reshape(-1, 1, 2)
        scale = calibrate_scale((0, 0, 60, 80), 10)
        result, _ = measure_fish(contour, scale)
        self.assertAlmostEqual(result["length_cm"], 20, places=4)
        self.assertAlmostEqual(result["width_cm"], 6, places=4)
        self.assertAlmostEqual(result["area_cm2"], 120, places=3)
        self.assertAlmostEqual(result["perimeter_cm"], 52, places=3)

    def test_dark_and_light_segmentation(self):
        for background, fish in [(240, 30), (30, 240)]:
            with self.subTest(background=background):
                image = np.full((240, 400, 3), background, np.uint8)
                cv2.rectangle(image, (90, 90), (290, 150), (fish,) * 3, -1)
                gray, _ = preprocess(image, blur_size=1)
                mask, _ = segment_fish(gray, morph_size=1)
                contour = find_fish_contour(mask)
                result, _ = measure_fish(contour, 10)
                self.assertAlmostEqual(result["length_cm"], 20)
                self.assertAlmostEqual(result["width_cm"], 6)

    def test_empty_and_clipped_masks(self):
        for mask in [np.zeros((100, 100), np.uint8), np.full((100, 100), 255, np.uint8)]:
            with self.assertRaises(ValueError):
                find_fish_contour(mask)

    def test_invalid_calibration(self):
        for line, cm in [((1, 1, 1, 1), 10), ((0, 0, 20, 0), 0), ((0, 0, 20, 0), float("nan"))]:
            with self.assertRaises(ValueError):
                calibrate_scale(line, cm)


if __name__ == "__main__":
    unittest.main()
