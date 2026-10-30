"""Giao diện Streamlit: python -m streamlit run app.py."""

import csv
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.main import (
    calibrate_scale, draw_result, find_fish_contour,
    measure_fish, preprocess, segment_fish,
)


def main():
    """Tải ảnh, chỉnh thông số và cập nhật kết quả ngay khi có thay đổi."""
    st.set_page_config(page_title="Đo kích thước cá Basa", page_icon="🐟", layout="wide")
    st.title("🐟 Đo kích thước cá Basa")
    st.caption("Ảnh tĩnh · OpenCV · Đo chiều dài, chiều rộng, diện tích và chu vi")

    with st.sidebar:
        st.header("1. Chọn ảnh")
        source = st.radio("Nguồn ảnh", ["Tải ảnh của bạn", "Ảnh demo"], key="source")
        uploaded = st.file_uploader("Kéo thả hoặc chọn ảnh cá", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"], key="upload")
        st.caption("Mỗi ảnh một con cá, nền trơn tương phản. Đặt vật tham chiếu cạnh cá.")

    is_demo = source == "Ảnh demo"
    if is_demo:
        demo = Path(__file__).parent / "data" / "demo.png"
        if not demo.exists():
            st.info("Chạy `python src/create_demo.py` để tạo ảnh demo, hoặc tải ảnh của bạn lên.")
            return
        data, filename = demo.read_bytes(), demo.name
    elif uploaded is not None:
        data, filename = uploaded.getvalue(), uploaded.name
    else:
        st.info("👈 Tải ảnh ở thanh bên trái hoặc chọn **Ảnh demo** để thử ngay.")
        st.markdown("**Cách sử dụng:** chọn ảnh → giới hạn vùng chứa cá → nhập tham chiếu → tải kết quả.")
        return

    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    except cv2.error:
        image = None
    if image is None:
        st.error("Không đọc được ảnh. Hãy chọn một tệp ảnh PNG/JPG hợp lệ.")
        return
    height, width = image.shape[:2]
    if min(height, width) < 3:
        st.error("Ảnh quá nhỏ để đo. Vui lòng chọn ảnh khác.")
        return
    # Mỗi ảnh có bộ điều khiển riêng, tránh giữ tọa độ của ảnh trước.
    image_key = hashlib.sha256(data).hexdigest()[:16] + ("demo" if is_demo else "upload")

    with st.sidebar:
        st.header("2. Vùng chứa cá")
        use_roi = st.checkbox("Giới hạn vùng đo (ROI)", value=is_demo, key=f"roi_{image_key}")
        x0, x1, y0, y1 = 0, width, 0, height
        if use_roi:
            st.caption("Chừa nền quanh toàn bộ cá; đưa thước và vật khác ra ngoài vùng đo.")
            x0, x1 = st.slider("Khoảng ngang X (pixel)", 0, width, (40, 840) if is_demo else (0, width), key=f"x_{image_key}")
            y0, y1 = st.slider("Khoảng dọc Y (pixel)", 0, height, (40, 440) if is_demo else (0, height), key=f"y_{image_key}")

        st.header("3. Hiệu chỉnh đơn vị")
        mode = st.selectbox("Cách hiệu chỉnh", ["Đoạn tham chiếu trên ảnh", "Tỷ lệ pixel/cm đã biết"], key="calibration")
        reference_line = None
        scale = None
        if mode == "Đoạn tham chiếu trên ảnh":
            st.caption("Nhập tọa độ hai đầu đoạn đã biết trên ảnh gốc. Đường vàng trên ảnh giúp kiểm tra vị trí.")
            defaults = (100, 520, 300, 520) if is_demo else (width // 4, height // 2, 3 * width // 4, height // 2)
            coords = []
            columns = st.columns(2)
            for i, label in enumerate(["X1", "Y1", "X2", "Y2"]):
                with columns[i % 2]:
                    coords.append(st.number_input(label, 0, width - 1 if i % 2 == 0 else height - 1, defaults[i], key=f"ref{i}_{image_key}"))
            reference_line = tuple(coords)
            reference_cm = st.number_input("Chiều dài thật của đoạn (cm)", min_value=0.01, value=10.0 if is_demo else None, step=0.1, key=f"cm_{image_key}")
            if reference_cm is not None:
                try:
                    scale = calibrate_scale(reference_line, reference_cm)
                except ValueError as error:
                    st.error(str(error))
        else:
            scale = st.number_input("Số pixel trên 1 cm", min_value=0.01, value=20.0 if is_demo else None, step=0.1, key=f"scale_{image_key}")
            st.caption("Chỉ dùng tỷ lệ cũ nếu camera, độ phân giải và khoảng cách đo không đổi.")

        with st.expander("Tinh chỉnh phân đoạn"):
            foreground_name = st.selectbox("Màu cá so với nền", ["Tự động", "Cá tối hơn nền", "Cá sáng hơn nền"])
            foreground = {"Tự động": "auto", "Cá tối hơn nền": "dark", "Cá sáng hơn nền": "light"}[foreground_name]
            blur = st.select_slider("Làm mờ Gaussian (1 = tắt)", [1, 3, 5, 7, 9, 11], value=5)
            morph = st.select_slider("Lọc hình thái (1 = tắt)", [1, 3, 5, 7, 9], value=3)
            clahe = st.checkbox("Tăng tương phản CLAHE")
            min_area = st.number_input("Diện tích tối thiểu (pixel²)", min_value=1, value=100)
        show_debug = st.checkbox("Xem ảnh trung gian", value=True)

    if is_demo:
        st.info("Đây là ảnh tổng hợp minh họa. Đoạn 200 pixel được quy ước dài 10 cm.")
    preview = image.copy()
    if use_roi:
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (255, 160, 0), 2)
    if reference_line is not None:
        a, b, c, d = reference_line
        cv2.line(preview, (a, b), (c, d), (0, 220, 255), 3)
        for point, label in [((a, b), "1"), ((c, d), "2")]:
            cv2.circle(preview, point, 5, (0, 0, 255), -1)
            cv2.putText(preview, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    left, right = st.columns(2)
    with left:
        st.subheader("Ảnh đầu vào")
        st.image(preview, channels="BGR", caption=f"{filename} · {width} × {height} pixel", width="stretch")
        st.caption("Gốc tọa độ ở góc trên trái; X tăng sang phải, Y tăng xuống dưới.")
    if x1 - x0 < 3 or y1 - y0 < 3:
        st.error("Vùng đo quá nhỏ. Hãy mở rộng hai khoảng X/Y.")
        return

    crop = image[y0:y1, x0:x1]
    gray, stages = preprocess(crop, blur, clahe)
    mask, segment_stages = segment_fish(gray, foreground, morph)
    stages.update(segment_stages)
    with right:
        st.subheader("Phân đoạn cá")
        st.image(mask, caption="Cá cần có màu trắng, nền màu đen.", width="stretch")
    if show_debug:
        with st.expander("Các bước xử lý ảnh", expanded=False):
            labels = ["Ảnh xám", "Tương phản", "Gaussian", "Ngưỡng Otsu", "Morphology"]
            tabs = st.tabs(labels)
            for tab, label, stage in zip(tabs, labels, stages.values()):
                with tab:
                    st.image(stage, caption=label, width="stretch")

    try:
        contour = find_fish_contour(mask, min_area)
        contour += np.array([[[x0, y0]]], dtype=np.int32)
        if scale is None:
            st.info("Nhập chiều dài thật của đoạn tham chiếu hoặc tỷ lệ pixel/cm ở thanh bên để nhận số đo cm.")
            return
        measurements, box = measure_fish(contour, scale)
        result = draw_result(image, contour, box, measurements, reference_line)
    except (ValueError, cv2.error) as error:
        st.error(f"Chưa đo được cá: {error}")
        st.caption("Kiểm tra mặt nạ, chỉnh vùng đo hoặc chọn lại màu cá so với nền.")
        return

    st.subheader("Kết quả đo")
    metrics = st.columns(4)
    for column, label, key, unit in zip(metrics, ["Chiều dài", "Chiều rộng", "Diện tích", "Chu vi"], ["length_cm", "width_cm", "area_cm2", "perimeter_cm"], ["cm", "cm", "cm²", "cm"]):
        column.metric(label, f"{measurements[key]:.2f} {unit}")
    st.caption(f"Tỷ lệ: {scale:.4f} pixel/cm. Kết quả tự cập nhật khi bạn thay ảnh hoặc thông số.")
    st.image(result, channels="BGR", width="stretch")
    st.caption("Chiều dài/rộng là hai cạnh hộp bao xoay, có tính vây/đuôi. Cần đối chiếu số đo tay; camera gần vuông góc và tham chiếu cùng mặt phẳng đo.")

    # Tải trực tiếp từ bộ nhớ; không ghi đè kết quả của lần chạy dòng lệnh.
    ok, encoded = cv2.imencode(".png", result)
    output = io.StringIO(newline="")
    row = {"image": filename, **{k: round(v, 6) for k, v in measurements.items()}}
    writer = csv.DictWriter(output, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    download_image, download_csv = st.columns(2)
    if ok:
        download_image.download_button("Tải ảnh kết quả PNG", encoded.tobytes(), file_name=f"{Path(filename).stem}_result.png", mime="image/png")
    download_csv.download_button("Tải bảng số đo CSV", output.getvalue().encode("utf-8-sig"), file_name=f"{Path(filename).stem}_measurements.csv", mime="text/csv")


if __name__ == "__main__":
    main()
