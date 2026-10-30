# Nhận diện và đo kích thước cá basa từ webcam

Dự án môn học Xử lý ảnh dùng Python và OpenCV để phát hiện cá basa đặt trên nền tương phản, ước lượng chiều dài/chiều rộng 2D từ camera laptop và lưu kết quả đo. Đây là prototype xử lý ảnh cổ điển, không phải ứng dụng web.

## Tính năng

- Đọc trực tiếp webcam laptop bằng `cv2.VideoCapture(0)`.
- Phân đoạn theo HSV hoặc ảnh xám/Otsu, lọc contour và hiển thị mask để hiệu chỉnh.
- Ước lượng chiều dài, chiều rộng, diện tích từ `minAreaRect` và contour.
- Hiệu chuẩn pixel/cm bằng cạnh marker vuông ArUco hoặc hai điểm trên thước.
- Lưu ảnh có chú thích và số đo trong CSV khi nhấn `S`.
- Tính MAE/MAPE từ CSV nhãn thực tế và kết quả dự đoán.

## Phạm vi và giả định

Phiên bản này đo vật thể đặt yên trên một mặt phẳng, camera cố định nhìn từ trên xuống, cá không bị che khuất và có nền tương phản. Marker/thước phải ở cùng mặt phẳng với cá. Thay đổi khoảng cách/góc camera sau hiệu chuẩn làm sai tỉ lệ. `minAreaRect` đo hộp xoay bao quanh contour nên là kích thước bao ngoài ước lượng, không phải phép đo giải phẫu hoặc 3D.

## Yêu cầu

- Windows, macOS hoặc Linux; Python 3.10–3.14 (khuyến nghị Python 3.14 nếu đã cài).
- Webcam tích hợp hoặc webcam tương thích OpenCV.
- Git (nếu đưa mã lên GitHub).

Thư viện được khai báo trong `requirements.txt`: OpenCV, NumPy và Matplotlib (Matplotlib chỉ dùng cho biểu đồ đánh giá).

## Cài đặt

Tại thư mục gốc dự án, tạo môi trường ảo:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__)"
```

Trên macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Chuẩn bị mặt phẳng đo và hiệu chuẩn

1. Cố định laptop/camera nhìn thẳng xuống bàn; đặt cá và vật chuẩn trên cùng mặt phẳng.
2. Dùng nền tương phản, ánh sáng tán xạ đều, hạn chế bóng và phản sáng.
3. Trong `config.json`, đặt `reference_type` là `aruco` hoặc `manual`.
4. Với `aruco`: in marker ArUco DICT_4X4_50 có ID được khai báo, đo chính xác cạnh in (mặc định 5 cm), đặt marker trong khung hình. Thư viện `opencv-contrib-python` hỗ trợ ArUco. Nếu không thấy marker, chương trình báo trạng thái chưa hiệu chuẩn.
5. Với `manual`: đặt `pixels_per_cm` bằng tỉ lệ đo được ở đúng vị trí mặt phẳng đặt cá. Ví dụ vật dài 10 cm chiếm 240 pixel thì nhập 24.0. Tỉ lệ thủ công không tự cập nhật.
6. Không di chuyển camera sau hiệu chuẩn. Kiểm tra ngưỡng phân đoạn bằng cửa sổ mask; sửa `segmentation_mode` và `hsv_lower`/`hsv_upper` trong cấu hình khi cần.

Marker ArUco phải được in đúng tỉ lệ, vuông, phẳng và không bị che. Phép đo dùng cạnh chiếu trong ảnh, do đó cần camera gần vuông góc với mặt phẳng. Phối cảnh mạnh sẽ gây sai số; có thể cải thiện bằng hiệu chỉnh homography của mặt phẳng trong phiên bản mở rộng.

## Chạy webcam

```bash
python src/main.py
```

- `S`: lưu ảnh hiện tại cùng kết quả vào `outputs/` và thêm một dòng vào `outputs/measurements.csv`.
- `Q` hoặc `Esc`: thoát.
- Cửa sổ `Mask` giúp xem vùng được phân đoạn.

Nếu không mở được camera, đóng ứng dụng khác đang sử dụng webcam, kiểm tra quyền camera hoặc đổi `camera_index` trong `config.json` (thử 1). Chỉ số mặc định là 0 theo yêu cầu dùng camera tích hợp.

### Khắc phục `ModuleNotFoundError: No module named 'cv2'`

Lỗi này nghĩa là OpenCV chưa được cài vào đúng Python đang chạy chương trình. Trong PowerShell, từ thư mục dự án, tạo môi trường ảo bằng Python 3.14 rồi cài requirements:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__)"
python src/main.py
```

Nếu không muốn kích hoạt môi trường, gọi trực tiếp đúng interpreter:

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 src/main.py
```

Không dùng `C:\Python315\python.exe` cho bản hiện tại: wheel OpenCV chính thức được phát hành cho các phiên bản Python đến 3.14, chưa gồm 3.15. Sau khi cài thư viện, hãy luôn chạy bằng Python của `.venv` hoặc `py -3.14`.

## Thu thập dữ liệu và đánh giá

Thu ít nhất 20–30 mẫu ban đầu; nên hướng đến 50–100 mẫu nếu điều kiện môn học cho phép. Chụp các mẫu trong nhiều tư thế và ánh sáng phù hợp với phạm vi dùng, ghi số đo thật bằng thước. Giữ riêng một phần mẫu đánh giá, không dùng chúng để chỉnh ngưỡng. Ghi file CSV theo định dạng `data/measurements.csv`:

```csv
filename,actual_length_cm,actual_width_cm
fish_001.jpg,31.2,7.4
fish_002.jpg,28.8,6.9
```

Sau khi có dự đoán từ chương trình webcam, ghép số đo thực tế với số đo dự đoán thành CSV có các cột dưới đây. Có thể dùng file mẫu `data/evaluation.csv`:

```bash
python src/evaluate.py --csv data/evaluation.csv
```

CSV đánh giá cần có các cột `filename,actual_length_cm,actual_width_cm,predicted_length_cm,predicted_width_cm`. Chương trình webcam lưu dự đoán theo thời gian trong `outputs/measurements.csv`; đối chiếu ảnh tương ứng, điền nhãn thật và chép số dự đoán vào các cột `predicted_...` của CSV đánh giá. Báo cáo MAE (cm) và MAPE (%) riêng cho chiều dài/rộng. MAPE không ổn định khi giá trị thực bằng 0; kích thước cá hợp lệ phải lớn hơn 0.

## Cấu trúc dự án

```text
.
├── config.json
├── data/
│   ├── evaluation.csv
│   ├── measurements.csv
│   ├── raw/.gitkeep
│   └── annotated/.gitkeep
├── outputs/.gitkeep
├── src/
│   ├── calibration.py
│   ├── evaluate.py
│   ├── measurement.py
│   ├── segmentation.py
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

Không commit môi trường ảo, ảnh/video riêng tư hoặc đầu ra thử nghiệm lớn. Một vài ảnh demo nhỏ có thể được thêm thủ công sau khi có sự đồng ý của người sở hữu dữ liệu.

## Luồng xử lý

```text
Webcam → làm mượt → HSV/Otsu → morphology → contour lớn nhất
       → marker ArUco hoặc tỉ lệ thủ công → kích thước 2D
       → overlay trực tiếp → nhấn S lưu ảnh/CSV
```

## Hạn chế và hướng phát triển

- Ngưỡng màu nhạy với màu cá, nền, cân bằng trắng và ánh sáng; xem mask và tinh chỉnh cấu hình.
- Contour có thể gồm vây, bóng hoặc vùng nền dính liền; diện tích chỉ loại được nhiễu đơn giản.
- Một tỉ lệ pixel/cm chỉ dùng ổn định khi camera và mặt phẳng không đổi.
- Có thể mở rộng bằng hiệu chỉnh phối cảnh/homography, chọn ROI, phân đoạn GrabCut, YOLO segmentation/Mask R-CNN, theo dõi video hoặc camera chiều sâu để đo 3D.

## Đưa lên GitHub

Tạo repository trống trên GitHub, rồi trong thư mục này:

```bash
git init
git add .
git commit -m "feat: add webcam fish measurement prototype"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

Commit theo thay đổi cụ thể, ví dụ `feat: add ArUco calibration`, `fix: reject small contours`, `docs: explain evaluation workflow`. Trang repo nên có tóm tắt bài toán, cách hiệu chuẩn, ảnh demo, bảng MAE/MAPE, điều kiện thử nghiệm và hạn chế.

