"""Kiểm tra và đánh giá toàn bộ tập ảnh cá trong data/train.

Cách chạy:
    python test_all_fish.py                   # Chạy tất cả 445 ảnh
    python test_all_fish.py --limit 10        # Thử nhanh 10 ảnh
    python test_all_fish.py --gt-only         # Chỉ test 118 ảnh có nhãn chuẩn Ground Truth
    python test_all_fish.py --save-viz        # Lưu ảnh đối chiếu trực quan (ảnh gốc + mask + GT)
    python test_all_fish.py --strict          # Bật chế độ nghiêm ngặt (báo lỗi nếu chạm viền)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Thiết lập encoding console Windows để hiển thị tiếng Việt mượt mà
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from src.main import calibrate_scale, measure_fish, preprocess, segment_fish


def find_contour_flexible(mask, min_area=50, allow_border=True):
    """Tìm contour cá lớn nhất, có tùy chọn cho phép hoặc từ chối chạm viền."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy contour nào trong ảnh mặt nạ")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        raise ValueError(f"Contour quá nhỏ ({cv2.contourArea(contour):.1f} px < {min_area} px)")
    x, y, w, h = cv2.boundingRect(contour)
    if not allow_border:
        if x <= 0 or y <= 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]:
            raise ValueError("Contour chạm sát viền ảnh (bật --allow-border nếu không dùng ROI)")
    return contour


def compute_metrics(pred_mask, gt_mask):
    """Tính toán chỉ số IoU (Jaccard Index) và Dice Score (F1)."""
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = (intersection / union) * 100.0 if union > 0 else 0.0

    sum_areas = pred_bin.sum() + gt_bin.sum()
    dice = (2.0 * intersection / sum_areas) * 100.0 if sum_areas > 0 else 0.0

    area_error = (
        abs(pred_bin.sum() - gt_bin.sum()) / gt_bin.sum() * 100.0
        if gt_bin.sum() > 0
        else 0.0
    )
    return iou, dice, area_error


def create_visualization(image, pred_mask, gt_mask, box, measurements, iou, dice):
    """Tạo ảnh ghép 3 khung hình: Ảnh gốc + Khung đo, Mặt nạ dự đoán, Mặt nạ Ground Truth."""
    h, w = image.shape[:2]

    # Khung 1: Ảnh gốc + Bounding box xoay
    vis1 = image.copy()
    if box is not None:
        cv2.polylines(vis1, [np.rint(box).astype(np.int32)], True, (0, 255, 0), 2)
        # Vẽ tâm và 2 trục dài/rộng
        midpoints = (box + np.roll(box, -1, axis=0)) / 2
        for a, b in [(midpoints[0], midpoints[2]), (midpoints[1], midpoints[3])]:
            cv2.line(vis1, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (0, 0, 255), 2)

    # Khung 2: Mặt nạ phân đoạn dự đoán (màu vàng)
    vis2 = np.zeros((h, w, 3), dtype=np.uint8)
    vis2[pred_mask > 0] = (0, 230, 255)

    # Khung 3: Mặt nạ Ground Truth chuẩn (màu xanh lá)
    vis3 = np.zeros((h, w, 3), dtype=np.uint8)
    if gt_mask is not None:
        vis3[gt_mask > 0] = (0, 255, 0)
    else:
        cv2.putText(vis3, "No Ground Truth", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

    # Ghép ngang 3 ảnh
    combined = np.hstack([vis1, vis2, vis3])

    # Thêm thanh thông tin phía trên
    bar_height = 45
    header = np.full((bar_height, combined.shape[1], 3), 40, dtype=np.uint8)
    text = (
        f"Length: {measurements.get('length_px', 0):.1f}px | "
        f"Width: {measurements.get('width_px', 0):.1f}px | "
        f"Area: {measurements.get('area_px', 0):.0f}px | "
    )
    if iou is not None:
        text += f"IoU: {iou:.1f}% | Dice: {dice:.1f}%"
    cv2.putText(header, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, combined])


def run_evaluation(args):
    data_dir = Path(args.data_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        (output_dir / "visuals").mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"❌ Lỗi: Không tìm thấy thư mục ảnh {data_dir}")
        return

    # Lấy danh sách ảnh
    image_paths = sorted(data_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(data_dir.glob("*.png"))

    if args.gt_only:
        image_paths = [p for p in image_paths if (gt_dir / f"{p.stem}.png").exists()]

    if args.limit:
        image_paths = image_paths[: args.limit]

    total_images = len(image_paths)
    print("=" * 70)
    print(f"🐟 BẮT ĐẦU KIỂM THỬ TẬP ẢNH CÁ ({total_images} ảnh)")
    print(f"📁 Thư mục ảnh: {data_dir}")
    print(f"🎯 Nhãn Ground Truth: {gt_dir}")
    print(f"⚙️  Chế độ viền: {'Nghiêm ngặt (Strict)' if args.strict else 'Linh hoạt (Allow border)'}")
    print("=" * 70)

    results = []
    success_count = 0
    failed_count = 0
    ious = []
    dices = []
    total_time_ms = 0.0

    start_total_time = time.perf_counter()

    for idx, img_path in enumerate(image_paths, 1):
        t0 = time.perf_counter()
        img = cv2.imread(str(img_path))
        if img is None:
            failed_count += 1
            results.append({
                "image": img_path.name,
                "status": "FAIL",
                "error": "Không đọc được file ảnh",
            })
            continue

        gt_path = gt_dir / f"{img_path.stem}.png"
        has_gt = gt_path.exists()
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if has_gt else None

        try:
            # Chạy pipeline xử lý ảnh
            gray, _ = preprocess(img, blur_size=args.blur, clahe=args.clahe, filter_type=args.filter_type)
            mask, _ = segment_fish(gray, foreground=args.foreground, morph_size=args.morph, auto_filter_ruler=not args.no_auto_ruler)
            contour = find_contour_flexible(mask, min_area=args.min_area, allow_border=not args.strict)

            # Đo kích thước hình học
            scale = args.pixels_per_cm if args.pixels_per_cm else 1.0
            measurements, box = measure_fish(contour, scale)

            # Tạo mặt nạ dự đoán từ contour
            pred_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(pred_mask, [contour], -1, 255, cv2.FILLED)

            iou, dice, area_error = None, None, None
            if has_gt and gt_mask is not None:
                iou, dice, area_error = compute_metrics(pred_mask, gt_mask)
                ious.append(iou)
                dices.append(dice)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_time_ms += elapsed_ms
            success_count += 1

            res_entry = {
                "image": img_path.name,
                "status": "OK",
                "length_px": round(measurements["length_px"], 2),
                "width_px": round(measurements["width_px"], 2),
                "area_px": round(measurements["area_cm2"] * (scale**2), 1),
                "perimeter_px": round(measurements["perimeter_cm"] * scale, 1),
                "has_gt": "Yes" if has_gt else "No",
                "iou_percent": round(iou, 2) if iou is not None else "",
                "dice_percent": round(dice, 2) if dice is not None else "",
                "area_error_percent": round(area_error, 2) if area_error is not None else "",
                "time_ms": round(elapsed_ms, 2),
                "error": "",
            }

            # Lưu ảnh trực quan nếu yêu cầu
            if args.save_viz:
                viz = create_visualization(img, pred_mask, gt_mask, box, res_entry, iou, dice)
                cv2.imwrite(str(output_dir / "visuals" / f"{img_path.stem}_eval.png"), viz)

            # In tiến trình từng dòng
            gt_info = f" | IoU: {iou:5.1f}% | Dice: {dice:5.1f}%" if iou is not None else ""
            print(f"[{idx:3d}/{total_images:3d}] ✅ {img_path.name:<15} -> Dài: {res_entry['length_px']:5.1f}px, Rộng: {res_entry['width_px']:5.1f}px{gt_info} ({elapsed_ms:4.1f}ms)")

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_time_ms += elapsed_ms
            failed_count += 1
            res_entry = {
                "image": img_path.name,
                "status": "FAIL",
                "length_px": "",
                "width_px": "",
                "area_px": "",
                "perimeter_px": "",
                "has_gt": "Yes" if has_gt else "No",
                "iou_percent": "",
                "dice_percent": "",
                "area_error_percent": "",
                "time_ms": round(elapsed_ms, 2),
                "error": str(e),
            }
            print(f"[{idx:3d}/{total_images:3d}] ❌ {img_path.name:<15} -> Lỗi: {e}")

        results.append(res_entry)

    total_wall_time = time.perf_counter() - start_total_time

    # Lưu kết quả ra file CSV
    csv_path = output_dir / "test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image", "status", "length_px", "width_px", "area_px",
            "perimeter_px", "has_gt", "iou_percent", "dice_percent",
            "area_error_percent", "time_ms", "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # In bảng tổng kết
    print("\n" + "=" * 70)
    print("📊 BÁO CÁO TỔNG HỢP KẾT QUẢ KIỂM THỬ")
    print("=" * 70)
    print(f"• Tổng số ảnh đã kiểm tra      : {total_images}")
    print(f"• Số ảnh xử lý thành công     : {success_count} ({success_count / total_images * 100:.1f}%)")
    print(f"• Số ảnh gặp lỗi               : {failed_count} ({failed_count / total_images * 100:.1f}%)")
    print(f"• Tổng thời gian chạy          : {total_wall_time:.2f} giây")
    print(f"• Tốc độ trung bình            : {total_images / total_wall_time:.1f} FPS ({total_time_ms / total_images:.1f} ms/ảnh)")

    if ious:
        print("-" * 70)
        print("🎯 ĐÁNH GIÁ ĐỘ CHÍNH XÁC PHÂN ĐOẠN (SO VỚI GROUND TRUTH):")
        print(f"• Số ảnh có Ground Truth      : {len(ious)} ảnh")
        print(f"• Chỉ số IoU trung bình (Jaccard) : {np.mean(ious):.2f}% (Cao nhất: {np.max(ious):.2f}%)")
        print(f"• Chỉ số Dice trung bình (F1)     : {np.mean(dices):.2f}% (Cao nhất: {np.max(dices):.2f}%)")
        good = sum(1 for x in ious if x >= 70.0)
        fair = sum(1 for x in ious if 50.0 <= x < 70.0)
        poor = sum(1 for x in ious if x < 50.0)
        print(f"  + Rất tốt (IoU >= 70%)       : {good} ảnh ({good / len(ious) * 100:.1f}%)")
        print(f"  + Khá     (50% <= IoU < 70%) : {fair} ảnh ({fair / len(ious) * 100:.1f}%)")
        print(f"  + Cần ROI (IoU < 50%)        : {poor} ảnh ({poor / len(ious) * 100:.1f}%)")

    print("-" * 70)
    print(f"💾 File kết quả CSV đã lưu tại : {csv_path}")
    if args.save_viz:
        print(f"🖼️  Ảnh trực quan đã lưu tại    : {output_dir / 'visuals'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Kiểm thử và đánh giá toàn bộ ảnh cá trong tập dữ liệu.")
    parser.add_argument("--data-dir", default="data/train/train_dist1/body", help="Thư mục chứa ảnh cá (.jpg)")
    parser.add_argument("--gt-dir", default="data/train/train_body_dist1/body", help="Thư mục chứa mask nhãn chuẩn (.png)")
    parser.add_argument("--output-dir", default="results/evaluation", help="Thư mục lưu kết quả CSV và ảnh")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ảnh kiểm tra (mặc định: kiểm tra hết)")
    parser.add_argument("--gt-only", action="store_true", help="Chỉ kiểm tra các ảnh có sẵn Ground Truth")
    parser.add_argument("--save-viz", action="store_true", help="Lưu ảnh đối chiếu trực quan (Ảnh + Dự đoán + Ground Truth)")
    parser.add_argument("--no-auto-ruler", action="store_true", help="Tắt tính năng tự động loại thước kẻ")
    parser.add_argument("--strict", action="store_true", help="Bắt buộc không được chạm viền (mặc định: linh hoạt)")
    parser.add_argument("--foreground", default="auto", choices=["auto", "dark", "light"], help="Cực tính cá so với nền")
    parser.add_argument("--filter-type", default="median", choices=["median", "gaussian"], help="Loại bộ lọc làm mịn (mặc định: median)")
    parser.add_argument("--blur", type=int, default=5, help="Kích thước Gaussian blur (mặc định: 5)")
    parser.add_argument("--morph", type=int, default=3, help="Kích thước Morphology (mặc định: 3)")
    parser.add_argument("--clahe", action="store_true", help="Bật cân bằng tương phản CLAHE")
    parser.add_argument("--min-area", type=int, default=50, help="Diện tích contour tối thiểu")
    parser.add_argument("--pixels-per-cm", type=float, default=None, help="Tỷ lệ pixel/cm nếu muốn quy đổi")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
