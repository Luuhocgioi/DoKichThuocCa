"""Giao diện Streamlit: python -m streamlit run app.py."""

import csv
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import importlib
import src.main
importlib.reload(src.main)
from src.main import (
    calibrate_scale, draw_result, find_fish_contour,
    measure_fish, preprocess, segment_fish,
)


def show_algorithm_tabs(crop, blur, clahe, foreground, morph, auto_filter_ruler,
                        min_area, allow_border, scale, offset, filter_type):
    """4 tab riêng: Gaussian, Median (bộ lọc) và Otsu cổ điển, Otsu đa cấp (ngưỡng)."""
    st.subheader("Xem từng thuật toán")
    st.caption("Bấm vào từng tab để xem riêng từng thuật toán.")
    tabs = st.tabs(["Gaussian", "Median", "Otsu cổ điển", "Otsu đa cấp"])

    for tab, ftype, title in ((tabs[0], "gaussian", "Lọc Gaussian"), (tabs[1], "median", "Lọc trung vị (Median)")):
        with tab:
            gray, stages = preprocess(crop, blur, clahe, filter_type=ftype)
            left, right = st.columns(2)
            left.image(stages["01_gray"], caption="Ảnh xám gốc", width="stretch")
            right.image(gray, caption=title, width="stretch")

    gray, _ = preprocess(crop, blur, clahe, filter_type=filter_type)
    for tab, method, title in ((tabs[2], "otsu", "Otsu cổ điển"), (tabs[3], "multi_otsu", "Otsu đa cấp")):
        with tab:
            st.caption(f"Dùng bộ lọc đang chọn ở thanh bên: {filter_type}.")
            mask, _ = segment_fish(gray, foreground, morph, auto_filter_ruler=auto_filter_ruler, method=method)
            st.image(mask, caption=f"Mặt nạ {title}", width="stretch")
            try:
                contour = find_fish_contour(mask, min_area, allow_border=allow_border)
                if scale is None:
                    st.info("Nhập tỷ lệ pixel/cm để xem số đo.")
                    continue
                contour = contour + np.array([[offset]], dtype=np.int32)
                m, _ = measure_fish(contour, scale)
                st.markdown(f"📏 **Dài:** `{m['length_cm']:.2f} cm` | **Rộng:** `{m['width_cm']:.2f} cm` | **Diện tích:** `{m['area_cm2']:.2f} cm²` | **Chu vi:** `{m['perimeter_cm']:.2f} cm`")
            except (ValueError, cv2.error) as error:
                st.warning(f"Chưa đo được bằng thuật toán này: {error}")


def set_quick_state(key, val):
    """Callback gán giá trị nhanh cho widget trước khi instantiate."""
    st.session_state[key] = val


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
            reference_cm = st.number_input(
                "Chiều dài thật của đoạn (cm)",
                min_value=0.01,
                value=10.0 if is_demo else None,
                step=0.1,
                key=f"cm_{image_key}",
                help="Gõ số rồi nhấn phím ENTER trên bàn phím để áp dụng"
            )
            st.caption("💡 *Mẹo: Gõ số xong hãy nhấn phím **Enter** (hoặc bấm nút 20 cm bên dưới)*")
            st.button(
                "⚡ Áp dụng nhanh thước chuẩn (20 cm)",
                key=f"quick_20_{image_key}",
                on_click=set_quick_state,
                args=(f"cm_{image_key}", 20.0),
                use_container_width=True,
                help="Tự động điền 20 cm để xem ngay kết quả đo mà không cần gõ phím"
            )
            if reference_cm is not None:
                try:
                    scale = calibrate_scale(reference_line, reference_cm)
                except ValueError as error:
                    st.error(str(error))
        else:
            scale = st.number_input("Số pixel trên 1 cm", min_value=0.01, value=20.0 if is_demo else None, step=0.1, key=f"scale_{image_key}")
            st.button(
                "⚡ Áp dụng nhanh tỷ lệ chuẩn (10 px/cm)",
                key=f"quick_scale_{image_key}",
                on_click=set_quick_state,
                args=(f"scale_{image_key}", 10.0),
                use_container_width=True,
                help="Tự động điền 10 px/cm để xem ngay kết quả đo"
            )
            st.caption("Chỉ dùng tỷ lệ cũ nếu camera, độ phân giải và khoảng cách đo không đổi.")

        st.header("4. Thuật toán xử lý")
        filter_options = ["Median (Lọc trung vị - Tối ưu cho cá thật)", "Gaussian (Lọc Gauss)"]
        filter_idx = 1 if is_demo else 0
        filter_choice = st.selectbox(
            "Bộ lọc khử nhiễu",
            filter_options,
            index=filter_idx,
            key=f"filter_{image_key}",
            help="Median khử nhiễu và bảo toàn sắc nét đường biên cá; Gaussian làm mờ mịn toàn bộ."
        )
        filter_type = "median" if "Median" in filter_choice else "gaussian"

        method_options = ["Otsu đa cấp (Multi-level Otsu - Tối ưu)", "Otsu cổ điển (Single Otsu)"]
        method_choice = st.selectbox(
            "Phương pháp phân ngưỡng",
            method_options,
            index=0,
            key=f"method_{image_key}",
            help="Multi-level Otsu tách sạch bóng đổ và khay nền xám; Otsu cổ điển chỉ dùng 1 ngưỡng."
        )
        threshold_method = "multi_otsu" if "Multi-level" in method_choice else "otsu"

        with st.expander("⚙️ Tinh chỉnh thông số nâng cao", expanded=False):
            foreground_name = st.selectbox("Màu cá so với nền", ["Tự động", "Cá tối hơn nền", "Cá sáng hơn nền"])
            foreground = {"Tự động": "auto", "Cá tối hơn nền": "dark", "Cá sáng hơn nền": "light"}[foreground_name]
            blur = st.select_slider("Kích thước bộ lọc (1 = tắt)", [1, 3, 5, 7, 9, 11], value=5)
            morph = st.select_slider("Lọc hình thái (1 = tắt)", [1, 3, 5, 7, 9], value=3)
            clahe = st.checkbox("Tăng tương phản CLAHE")
            min_area = st.number_input("Diện tích tối thiểu (pixel²)", min_value=1, value=100)
            auto_filter_ruler = st.checkbox("Tự động loại bỏ thước kẻ dính liền", value=True, help="Tự động phát hiện và ngắt dải thước kẻ dính vào thân cá")
            allow_border = st.checkbox("Cho phép cá chạm viền ảnh", value=True, help="Bật nếu đuôi hoặc mõm cá nằm sát mép ảnh/ROI")

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
    gray, stages = preprocess(crop, blur, clahe, filter_type=filter_type)
    mask, segment_stages = segment_fish(gray, foreground, morph, auto_filter_ruler=auto_filter_ruler, method=threshold_method)
    stages.update(segment_stages)
    with right:
        st.subheader("Phân đoạn cá")
        st.image(mask, caption="Cá cần có màu trắng, nền màu đen.", width="stretch")
    if show_debug:
        with st.expander("Các bước xử lý ảnh (Chi tiết từng bước thuật toán)", expanded=True):
            friendly_names = {
                "01_gray": "1. Ảnh xám",
                "02_contrast": "2. Tương phản (CLAHE)",
                "03_gaussian_blur": "3. Lọc Gaussian",
                "03_median_blur": "3. Lọc trung vị (Median)",
                "03_no_blur": "3. Không làm mờ",
                "04_threshold": "4. Ngưỡng Otsu",
                "04_otsu_single": "4. Otsu Cấp 1 (Sơ bộ)" if "04b_multi_otsu" in stages else "4. Ngưỡng Otsu",
                "04b_multi_otsu": "5. Otsu Cấp 2 (Multi-level khử bóng)",
                "05_morphology": "6. Lọc hình thái (Morphology)" if "04b_multi_otsu" in stages else "5. Lọc hình thái (Morphology)",
                "06_auto_ruler_filter": "7. Tách thước kẻ dính liền" if "04b_multi_otsu" in stages else "6. Tách thước kẻ dính liền",
            }
            labels = [friendly_names.get(k, k) for k in stages.keys()]
            tabs = st.tabs(labels)
            for tab, (stage_key, stage_img) in zip(tabs, stages.items()):
                title = friendly_names.get(stage_key, stage_key)
                with tab:
                    st.image(stage_img, caption=title, width="stretch")

    show_algorithm_tabs(crop, blur, clahe, foreground, morph, auto_filter_ruler,
                        min_area, allow_border, scale, (x0, y0), filter_type)

    try:
        contour = find_fish_contour(mask, min_area, allow_border=allow_border)
        contour += np.array([[[x0, y0]]], dtype=np.int32)
        if scale is None:
            st.info("👉 Bạn chưa nhập chiều dài tham chiếu để quy đổi kích thước sang centimet.")
            st.button(
                "⚡ Nhấn vào đây để áp dụng nhanh thước chuẩn 20 cm (Xem kết quả đo ngay)",
                key=f"quick_main_{image_key}",
                on_click=set_quick_state,
                args=(f"cm_{image_key}", 20.0),
                use_container_width=True
            )
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
