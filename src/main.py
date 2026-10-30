"""Real-time basa fish measurement from the laptop webcam."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2

from calibration import ReferenceCalibrator
from measurement import measure_contour
from segmentation import segment


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def append_measurement(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    fields = [
        "timestamp", "length_cm", "width_cm", "area_cm2",
        "pixels_per_cm", "reference_type", "image",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    config = load_config()
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_index = int(config.get("camera_index", 0))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Không mở được webcam index {camera_index}. "
            "Kiểm tra quyền camera, đóng ứng dụng đang dùng webcam hoặc đổi camera_index."
        )

    calibrator = ReferenceCalibrator(config)
    cv2.namedWindow("Basa fish measurement", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Segmentation mask", cv2.WINDOW_NORMAL)
    last_result = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Không đọc được khung hình từ webcam.")
                break

            display = frame.copy()
            mask, contour = segment(frame, config)
            scale, reference_label = calibrator.get_scale(frame)
            last_result = None

            if contour is not None:
                result = measure_contour(contour, scale)
                last_result = result
                cv2.drawContours(display, [contour], -1, (255, 140, 0), 2)
                cv2.drawContours(display, [result["box"]], 0, (0, 255, 0), 2)

                if scale is not None:
                    label = (
                        f"L={result['length_cm']:.1f} cm  "
                        f"W={result['width_cm']:.1f} cm"
                    )
                    cv2.putText(
                        display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        display, f"Area={result['area_cm2']:.1f} cm2",
                        (20, 72), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        display, "Chua hieu chuan: dat marker/thiet lap scale",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 165, 255), 2, cv2.LINE_AA,
                    )

            status = f"Scale: {reference_label} | S: save | Q/Esc: quit"
            cv2.putText(
                display, status, (20, display.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                2, cv2.LINE_AA,
            )
            cv2.imshow("Basa fish measurement", display)
            if config.get("show_mask", True):
                cv2.imshow("Segmentation mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if last_result is None:
                    print("Chưa tìm thấy contour đủ lớn để lưu.")
                    continue
                if scale is None:
                    print("Chưa có tỉ lệ pixel/cm; chưa thể lưu số đo cm.")
                    continue

                timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                filename = f"measurement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = output_dir / filename
                if not cv2.imwrite(str(image_path), display):
                    print("Không ghi được ảnh đầu ra.")
                    continue
                append_measurement(
                    output_dir / "measurements.csv",
                    {
                        "timestamp": timestamp,
                        "length_cm": f"{last_result['length_cm']:.3f}",
                        "width_cm": f"{last_result['width_cm']:.3f}",
                        "area_cm2": f"{last_result['area_cm2']:.3f}",
                        "pixels_per_cm": f"{scale:.4f}",
                        "reference_type": reference_label,
                        "image": filename,
                    },
                )
                print(f"Đã lưu {image_path.name} và số đo vào measurements.csv")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

