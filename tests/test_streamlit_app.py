"""Kiểm tra tải ảnh và thay đổi trạng thái giao diện Streamlit."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).resolve().parents[1] / "app.py")


class StreamlitTests(unittest.TestCase):
    def test_demo_and_invalid_reference(self):
        app = AppTest.from_file(APP).run()
        self.assertFalse(app.exception)
        app.radio(key="source").set_value("Ảnh demo").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(app.metric[0].value, "28.93 cm")
        next(n for n in app.number_input if n.label == "X2").set_value(100).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)

    def test_upload_requires_calibration(self):
        image = np.full((240, 400, 3), 240, np.uint8)
        cv2.rectangle(image, (90, 90), (290, 150), (30, 30, 30), -1)
        _, encoded = cv2.imencode(".png", image)
        uploaded = SimpleNamespace(name="fish.png", getvalue=lambda: encoded.tobytes())
        with patch("streamlit.file_uploader", return_value=uploaded):
            app = AppTest.from_file(APP).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 0)
            app.selectbox(key="calibration").set_value("Tỷ lệ pixel/cm đã biết").run()
            next(n for n in app.number_input if n.label == "Số pixel trên 1 cm").set_value(10.0).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 4)
            self.assertAlmostEqual(float(app.metric[0].value.split()[0]), 20, delta=0.2)
            app.checkbox[0].check().run()
            next(s for s in app.slider if s.label == "Khoảng ngang X (pixel)").set_value((100, 101)).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(len(app.metric), 0)

    def test_corrupt_upload(self):
        uploaded = SimpleNamespace(name="bad.png", getvalue=lambda: b"not an image")
        with patch("streamlit.file_uploader", return_value=uploaded):
            app = AppTest.from_file(APP).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)


if __name__ == "__main__":
    unittest.main()
