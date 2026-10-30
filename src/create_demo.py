"""Tạo ảnh minh họa tổng hợp, không thay thế ảnh cá Basa thật."""

from pathlib import Path

import cv2
import numpy as np

try:
    from .main import configure_console, save_image
except ImportError:
    from main import configure_console, save_image


def main():
    configure_console()
    image = np.full((600, 900, 3), 235, dtype=np.uint8)
    cv2.ellipse(image, (430, 260), (240, 75), -12, 0, 360, (65, 70, 75), -1)
    tail = np.array([[630, 215], [750, 135], [770, 270]], dtype=np.int32)
    cv2.fillPoly(image, [tail], (65, 70, 75))
    cv2.circle(image, (230, 285), 7, (15, 15, 15), -1)
    # Hai vạch x=100 và x=300 cách nhau 200 px, giả định tương ứng 10 cm.
    cv2.line(image, (100, 520), (300, 520), (20, 20, 20), 2)
    for x in range(100, 301, 20):
        cv2.line(image, (x, 510), (x, 530), (20, 20, 20), 2)
    cv2.putText(image, "Reference: 10 cm (synthetic)", (100, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    path = Path(__file__).resolve().parents[1] / "data" / "demo.png"
    save_image(path, image)
    print(f"Đã tạo ảnh tổng hợp: {path}")


if __name__ == "__main__":
    main()
