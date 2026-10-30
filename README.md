# Đo kích thước cá Basa từ ảnh tĩnh

Đồ án Xử lý ảnh & Nhận dạng, giai đoạn 1: Python + OpenCV + NumPy,
không dùng deep learning. Mỗi ảnh chứa **một con cá** trên nền tương phản.
Chương trình tự phân đoạn và đo sau khi người dùng cung cấp tỷ lệ tham chiếu.

## Cấu trúc

```text
data/                 Ảnh đầu vào; create_demo.py tạo ảnh tổng hợp ở đây
src/main.py           Pipeline và giao diện dòng lệnh
src/create_demo.py    Tạo hình minh họa để thử nhanh
tests/                Kiểm tra hình học và phân đoạn cơ bản
results/              Ảnh kết quả, CSV và ảnh trung gian (không đưa vào Git)
requirements.txt      Thư viện cần cài
PROMPT.md             Yêu cầu gốc; không tự ý sửa nội dung cốt lõi
PROGRESS.md           Trạng thái và bước tiếp theo của mỗi phiên
```

## Cài đặt và chạy thử (PowerShell)

### Giao diện Streamlit (kéo thả ảnh)

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở http://localhost:8501 trong trình duyệt. Chọn **Ảnh demo** để thử ngay,
hoặc kéo thả ảnh vào thanh bên. Bật ROI nếu cần loại thước khỏi vùng đo.
Nhập hai tọa độ đầu đoạn tham chiếu và chiều dài thật (cm), hoặc nhập tỷ lệ
pixel/cm đã biết. Đường vàng trên ảnh cho biết đoạn tham chiếu đã chọn.
Giao diện hiện hỗ trợ nhập tọa độ, chưa hỗ trợ bấm điểm trực tiếp trên ảnh.

Kết quả và mặt nạ tự cập nhật khi thay thông số. Có nút tải PNG và CSV;
các tệp này được tải qua trình duyệt, không tự lưu vào `results/`.
Với ảnh tải lên, ứng dụng chưa xuất số đo cm cho đến khi nhập hiệu chỉnh.
Mở **Tinh chỉnh phân đoạn** để đổi cực tính, Gaussian, morphology, CLAHE.
Ảnh trung gian vẫn xem được khi chưa hiệu chỉnh hoặc không tìm thấy contour.
Dừng máy chủ bằng Ctrl+C trong terminal.

### Chạy bằng dòng lệnh

Chạy từ thư mục gốc dự án. Dùng Python 3.10 trở lên; phiên khởi tạo đã chạy
với Python 3.14, OpenCV 4.13.0 và NumPy 2.4.4.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/create_demo.py
.\.venv\Scripts\python.exe src/main.py data/demo.png --reference-line 100 520 300 520 --reference-cm 10 --roi 40 40 800 400 --debug
```

Nếu đã cài thư viện trong Python hiện tại, thay `.\.venv\Scripts\python.exe`
bằng `python`. Ảnh demo là hình tổng hợp có dáng cá, **không phải ảnh cá thật**.
Đoạn tham chiếu dài 200 pixel được gán 10 cm → tỷ lệ 20 pixel/cm.

Kết quả:

- `results/demo.png/result.png`: contour xanh lá, hộp xoay xanh dương,
  hai trục đo đỏ, đoạn tham chiếu vàng và bảng kích thước.
- `results/demo.png/debug/`: ảnh gốc, ROI, xám, tương phản, mờ, ngưỡng,
  morphology và contour được chọn.
- `results/measurements.csv`: một dòng mỗi ảnh, kích thước cm, diện tích cm²,
  chu vi cm, tỷ lệ pixel/cm, đường dẫn kết quả và trạng thái/lỗi.

Mỗi lần chạy ghi lại CSV và ảnh cùng tên trong thư mục đầu ra. Để giữ từng
lần thử, dùng `--output results/thu_01`, `--output results/thu_02`, v.v.
Khi một ảnh lỗi, chương trình vẫn xử lý các ảnh còn lại và trả mã thoát 1;
chỉ đọc các dòng CSV có `status=ok` (ảnh từ lần chạy cũ có thể còn trên đĩa).

## Chạy với ảnh thật và hiệu chỉnh

Đặt thước hoặc vật có kích thước đã biết cạnh cá. Chụp camera gần vuông góc
với mặt phẳng đo; vật tham chiếu cần cùng mặt phẳng với phần cá cần đo.
Giữ trọn cá trong ảnh, cá tương đối thẳng, nền trơn và ánh sáng đều.

Chọn hai đầu đoạn đã biết trên vật tham chiếu bằng chuột:

```powershell
python src/main.py data/ca_01.jpg --select-reference --reference-cm 10 --debug --show
```

Bấm chính xác hai vạch cách nhau 10 cm; nhấn Esc nếu cần hủy và chạy lại.
`--select-reference` cần máy có giao diện đồ họa và gói `opencv-python`.
Nếu dùng cạnh A4, cạnh ngắn là 21 cm, cạnh dài 29.7 cm; chọn đúng hai đầu
của một cạnh. Không tự suy ra kích thước thật chỉ từ ảnh không có tham chiếu.

Nếu vật tham chiếu hoặc vật khác có thể bị nhận nhầm, thêm
`--roi X Y W H` để vùng xử lý chỉ chứa cá và một khoảng nền xung quanh.
X, Y là góc trái trên; W, H là rộng và cao theo pixel. Tọa độ ROI và đoạn
tham chiếu luôn tính trên **ảnh gốc**, không phải ảnh đã thu nhỏ khi hiển thị.
Không cắt vây/đuôi. Đoạn tham chiếu được phép nằm ngoài ROI nhưng phải trong ảnh.

Có thể nhập tọa độ thay vì bấm chuột:

```powershell
python src/main.py data/ca_01.jpg --reference-line 100 520 300 520 --reference-cm 10 --roi 40 40 800 400 --debug
```

Các tọa độ trên chỉ là ví dụ cho ảnh demo; cần thay bằng tọa độ ảnh thực tế.
Tỷ lệ được tính bằng khoảng cách Euclid hai điểm chia chiều dài thật:

```text
pixel_per_cm = sqrt((x2-x1)^2 + (y2-y1)^2) / reference_cm
length_cm = length_px / pixel_per_cm
area_cm2 = contour_area_px / pixel_per_cm^2
```

Xử lý nhiều ảnh, chọn tham chiếu riêng cho từng ảnh:

```powershell
python src/main.py data --select-reference --reference-cm 10 --output results/batch --debug
```

Hoặc dùng tỷ lệ đã hiệu chỉnh khi camera, độ phân giải, khoảng cách đến
mặt phẳng đo không thay đổi:

```powershell
python src/main.py data --pixels-per-cm 20 --output results/batch --debug
```

Thư mục chỉ được quét một cấp. `--roi` và `--reference-line` dùng chung cho
mọi ảnh trong một lần chạy; chỉ dùng khi bố cục tương ứng không đổi.
Đổi cm sang mm bằng cách nhân kích thước/chu vi với 10, diện tích với 100.

## Phương pháp và debug

| Bước | Lý do chọn | Cách kiểm tra/điều chỉnh |
| --- | --- | --- |
| Chuyển xám | Đơn giản với nền có độ sáng khác cá | `01_gray.png` |
| CLAHE | Tùy chọn tăng tương phản cục bộ | Mặc định tắt; bật `--clahe`; xem `02_contrast.png` |
| Gaussian | Giảm nhiễu trước ngưỡng | `--blur 5`; `--blur 1` để tắt |
| Otsu | Tự chọn ngưỡng sáng/tối | `04_threshold.png`; cá phải trắng, nền đen |
| Chọn cực tính | Auto coi đa số viền ROI là nền | Ép `--foreground dark` hoặc `light` nếu auto chọn sai |
| Morphology | Opening bỏ đốm nhỏ, closing nối khe nhỏ | `--morph 3`; `--morph 1` để tắt; xem `05_morphology.png` |
| Contour lớn nhất | Dễ hiểu, phù hợp ảnh một cá đã cô lập | `06_selected_contour.png`; lọc vùng nhỏ với `--min-area 100` |
| minAreaRect | Đo hộp bao xoay theo hướng cá | Xem hộp và hai trục trên ảnh kết quả |

Kernel blur/morph phải là số lẻ dương. `--debug` bật lưu ảnh trung gian;
`--show` bật cửa sổ ảnh đầu vào/kết quả. Các bước bắt buộc như chuyển xám,
ngưỡng và contour được kiểm tra bằng ảnh trung gian, không có nút tắt.
Khi contour chạm viền, chương trình báo lỗi thay vì xuất kích thước cá bị cắt.

**Ý nghĩa phép đo:** chiều dài là cạnh dài, chiều rộng là cạnh ngắn của
hình chữ nhật xoay bao silhouette. Đây chưa phải phép đo chiều dài dọc
xương sống hoặc chiều rộng thân không tính vây. Diện tích là diện tích
hình chiếu theo contour ngoài; chu vi là chu vi contour, nhạy với nhiễu.
Không đo diện tích bề mặt 3D hoặc khối lượng.

Giới hạn: nền phức tạp, bóng đổ, phản sáng, cá cong, vây trong suốt và
nhiều vật thể có thể gây phân đoạn sai. Contour lớn nhất không xác minh
đó là cá. CLAHE hoặc morphology quá mạnh cũng có thể làm sai biên.
Hiệu chỉnh một tỷ lệ chưa xử lý phối cảnh, méo ống kính hay độ cao thân cá.
Cần kiểm tra mask và đối chiếu thước đo thật trước khi kết luận độ chính xác.

## Kiểm tra mã nguồn

```powershell
python -m unittest discover -s tests -v
```

Kiểm tra hình chữ nhật xoay có kích thước biết trước, quy đổi diện tích/
chu vi, nền sáng/tối, ảnh rỗng, contour bị cắt và tham chiếu không hợp lệ.
Các kiểm tra tổng hợp không thay cho đánh giá trên bộ ảnh cá thật.

## Quy trình phiên tiếp theo

Đọc `PROMPT.md` và `PROGRESS.md`, tóm tắt trạng thái, tiếp tục bước đề xuất
và cập nhật tiến trình khi kết thúc. Cần người dùng cung cấp ảnh cá thật,
độ dài đoạn tham chiếu, chiều dài/rộng đo tay và quy ước có tính vây/đuôi
hay không để tinh chỉnh và tính sai số.

