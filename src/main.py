"""Đo kích thước hình chiếu của một con cá trên ảnh tĩnh."""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def configure_console():
    """Giữ thông báo tiếng Việt khi Windows chuyển đầu ra qua pipe/file."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load_image(path):
    """Đọc ảnh màu, hỗ trợ đường dẫn có dấu trên Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def save_image(path, image):
    """Lưu ảnh PNG và tạo thư mục nếu chưa tồn tại."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Không mã hóa được ảnh: {path}")
    encoded.tofile(str(path))


def preprocess(image, blur_size=5, clahe=False):
    """Chuyển xám, tùy chọn CLAHE, rồi Gaussian để giảm nhiễu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray) if clahe else gray.copy()
    blurred = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0) if blur_size > 1 else enhanced.copy()
    return blurred, {"01_gray": gray, "02_contrast": enhanced, "03_blur": blurred}


def segment_fish(gray, foreground="auto", morph_size=3):
    """Otsu tách sáng/tối; chế độ auto coi đa số viền ảnh là nền."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    if foreground == "dark" or (foreground == "auto" and np.mean(border) > 127):
        mask = cv2.bitwise_not(mask)
    raw = mask.copy()
    if morph_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, {"04_threshold": raw, "05_morphology": mask}


def find_fish_contour(mask, min_area=100):
    """Lấy contour ngoài lớn nhất; báo lỗi nếu quá nhỏ hoặc bị cắt ở viền."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy cá. Kiểm tra ảnh ngưỡng và màu nền.")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        raise ValueError("Vùng tìm được quá nhỏ. Kiểm tra phân đoạn hoặc --min-area.")
    x, y, w, h = cv2.boundingRect(contour)
    if x == 0 or y == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]:
        raise ValueError("Contour chạm viền ảnh/ROI: hãy chừa nền quanh toàn bộ cá hoặc sửa phân đoạn.")
    return contour


def calibrate_scale(reference_line, reference_cm):
    """Tính pixel/cm từ hai đầu đoạn tham chiếu có độ dài thật biết trước."""
    if not math.isfinite(reference_cm) or reference_cm <= 0:
        raise ValueError("Chiều dài tham chiếu phải hữu hạn và lớn hơn 0 cm.")
    x1, y1, x2, y2 = reference_line
    pixels = math.hypot(x2 - x1, y2 - y1)
    if not math.isfinite(pixels) or pixels < 1:
        raise ValueError("Hai điểm tham chiếu phải cách nhau ít nhất 1 pixel.")
    return pixels / reference_cm


def measure_fish(contour, pixels_per_cm):
    """Đo hình chữ nhật xoay bao cá và diện tích/chu vi contour."""
    if not math.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
        raise ValueError("Tỷ lệ pixel/cm phải hữu hạn và lớn hơn 0.")
    rect = cv2.minAreaRect(contour)
    length, width = sorted(rect[1], reverse=True)
    if width <= 0:
        raise ValueError("Contour không đủ diện tích để đo.")
    measurements = {
        "pixels_per_cm": pixels_per_cm,
        "length_px": length,
        "width_px": width,
        "length_cm": length / pixels_per_cm,
        "width_cm": width / pixels_per_cm,
        "area_cm2": cv2.contourArea(contour) / pixels_per_cm ** 2,
        "perimeter_cm": cv2.arcLength(contour, True) / pixels_per_cm,
    }
    return measurements, cv2.boxPoints(rect)


def draw_result(image, contour, box, measurements, reference_line=None):
    """Vẽ contour, hộp xoay, hai trục đo và nhãn lên ảnh gốc."""
    result = image.copy()
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
    cv2.polylines(result, [np.rint(box).astype(np.int32)], True, (255, 160, 0), 2)
    midpoints = (box + np.roll(box, -1, axis=0)) / 2
    for a, b in [(midpoints[0], midpoints[2]), (midpoints[1], midpoints[3])]:
        cv2.line(result, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (0, 0, 255), 2)
    if reference_line is not None:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), reference_line)
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # Font mặc định OpenCV không hỗ trợ dấu tiếng Việt: nhãn dùng ASCII.
    labels = [
        f"Length: {measurements['length_cm']:.2f} cm",
        f"Width: {measurements['width_cm']:.2f} cm",
        f"Area: {measurements['area_cm2']:.2f} cm^2",
        f"Perimeter: {measurements['perimeter_cm']:.2f} cm",
    ]
    # Thêm vùng chú thích riêng để không che cá hoặc cắt nhãn trên ảnh nhỏ.
    canvas = np.full((result.shape[0] + 120, max(result.shape[1], 440), 3), 255, np.uint8)
    canvas[120:, :result.shape[1]] = result
    for i, label in enumerate(labels):
        cv2.putText(canvas, label, (10, 25 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    return canvas


def select_reference(image):
    """Cho người dùng bấm hai đầu đoạn tham chiếu; Esc để hủy."""
    points = []
    factor = min(1.0, 1200 / image.shape[1], 800 / image.shape[0])
    preview = cv2.resize(image, None, fx=factor, fy=factor)
    scale_x, scale_y = image.shape[1] / preview.shape[1], image.shape[0] / preview.shape[0]
    title = "Click 2 reference endpoints - Esc: cancel"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x * scale_x, y * scale_y))
            cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    try:
        while len(points) < 2:
            cv2.imshow(title, preview)
            if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Đã hủy chọn tham chiếu.")
    finally:
        cv2.destroyAllWindows()
    return (*points[0], *points[1])


def process_image(path, args, output_dir):
    """Chạy pipeline một ảnh; mọi tọa độ đầu vào tính trên ảnh gốc."""
    image = load_image(path)
    if args.debug:
        save_image(output_dir / "debug" / "00_input.png", image)
    reference_line = select_reference(image) if args.select_reference else args.reference_line
    if reference_line is not None:
        for x, y in np.asarray(reference_line).reshape(2, 2):
            if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                raise ValueError("Điểm tham chiếu nằm ngoài ảnh.")
        scale = calibrate_scale(reference_line, args.reference_cm)
    else:
        scale = args.pixels_per_cm
    x, y, w, h = args.roi or (0, 0, image.shape[1], image.shape[0])
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError("ROI không hợp lệ hoặc nằm ngoài ảnh.")
    crop = image[y:y + h, x:x + w]
    gray, stages = preprocess(crop, args.blur, args.clahe)
    mask, segment_stages = segment_fish(gray, args.foreground, args.morph)
    stages.update(segment_stages)
    if args.debug:
        save_image(output_dir / "debug" / "00_roi.png", crop)
        for name, stage in stages.items():
            save_image(output_dir / "debug" / f"{name}.png", stage)
    contour = find_fish_contour(mask, args.min_area)
    contour = contour + np.array([[[x, y]]], dtype=np.int32)
    measurements, box = measure_fish(contour, scale)
    result = draw_result(image, contour, box, measurements, reference_line)
    save_image(output_dir / "result.png", result)
    if args.debug:
        selected = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(selected, [contour], -1, 255, cv2.FILLED)
        save_image(output_dir / "debug" / "06_selected_contour.png", selected)
    if args.show:
        cv2.imshow("Input", image)
        cv2.imshow("Fish measurement - press any key", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return measurements


def main():
    """Nhận ảnh hoặc thư mục, xử lý từng ảnh và xuất chung một bảng CSV."""
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Đường dẫn ảnh hoặc thư mục ảnh")
    parser.add_argument("--output", type=Path, default=Path("results"))
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--reference-line", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    calibration.add_argument("--select-reference", action="store_true", help="Bấm hai điểm trên từng ảnh")
    calibration.add_argument("--pixels-per-cm", type=float, help="Tỷ lệ đã hiệu chỉnh từ trước")
    parser.add_argument("--reference-cm", type=float, help="Chiều dài thật của đoạn tham chiếu (cm)")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Vùng chỉ chứa cá")
    parser.add_argument("--foreground", choices=["auto", "dark", "light"], default="auto")
    parser.add_argument("--blur", type=int, default=5, help="Kernel Gaussian lẻ; 1 = tắt")
    parser.add_argument("--morph", type=int, default=3, help="Kernel morphology lẻ; 1 = tắt")
    parser.add_argument("--clahe", action="store_true", help="Bật cân bằng tương phản cục bộ")
    parser.add_argument("--min-area", type=float, default=100, help="Diện tích nhỏ nhất (pixel vuông)")
    parser.add_argument("--debug", action="store_true", help="Lưu ảnh trung gian")
    parser.add_argument("--show", action="store_true", help="Hiển thị ảnh bằng cửa sổ OpenCV")
    args = parser.parse_args()
    for name in ("blur", "morph"):
        value = getattr(args, name)
        if value < 1 or value % 2 == 0:
            parser.error(f"--{name} phải là số lẻ dương (1 để tắt).")
    if not math.isfinite(args.min_area) or args.min_area <= 0:
        parser.error("--min-area phải hữu hạn và lớn hơn 0.")
    if args.pixels_per_cm is not None:
        if not math.isfinite(args.pixels_per_cm) or args.pixels_per_cm <= 0:
            parser.error("--pixels-per-cm phải hữu hạn và lớn hơn 0.")
        if args.reference_cm is not None:
            parser.error("Không dùng --reference-cm cùng --pixels-per-cm.")
    elif args.reference_cm is None or not math.isfinite(args.reference_cm) or args.reference_cm <= 0:
        parser.error("Cần --reference-cm hữu hạn và lớn hơn 0 khi chọn đoạn tham chiếu.")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error("Thư mục không có ảnh được hỗ trợ.")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["image", "status", "error", "result_image", "pixels_per_cm", "length_px", "width_px", "length_cm", "width_cm", "area_cm2", "perimeter_cm"]
    failed = 0
    with (args.output / "measurements.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            # Giữ phần mở rộng để a.jpg và a.png không ghi đè nhau.
            output_dir = args.output / path.name
            row = {"image": str(path), "status": "error", "error": ""}
            try:
                measurement = process_image(path, args, output_dir)
                row.update({k: round(v, 6) for k, v in measurement.items()})
                row.update(status="ok", result_image=str(output_dir / "result.png"))
                print(f"{path.name}: dài {measurement['length_cm']:.2f} cm; rộng {measurement['width_cm']:.2f} cm")
            except (ValueError, OSError, cv2.error) as error:
                row["error"] = str(error)
                failed += 1
                print(f"Lỗi {path.name}: {error}")
            writer.writerow(row)
            handle.flush()
    print(f"Đã lưu CSV: {args.output / 'measurements.csv'} ({len(paths) - failed}/{len(paths)} ảnh thành công)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
