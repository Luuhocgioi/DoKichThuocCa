# Tiến trình đồ án

- Ngày cập nhật gần nhất: 2026-10-30 (Asia/Bangkok).
- Trạng thái ban đầu: **Chưa bắt đầu**.
- Trạng thái hiện tại: **Hoàn thành bản triển khai giai đoạn 1; chờ kiểm chứng trên ảnh cá thật**.
- Cập nhật cùng ngày: đã bổ sung giao diện Streamlit tải ảnh theo yêu cầu mới.

## Đã hoàn thành
- [x] Lưu nguyên nội dung yêu cầu vào PROMPT.md.
- [x] Khởi tạo data/, src/, results/, tests/, requirements.txt, README.md, .gitignore.
- [x] Viết load_image, preprocess, segment_fish, find_fish_contour, calibrate_scale, measure_fish, draw_result, main.
- [x] Chuyển xám, Gaussian, tùy chọn CLAHE; Otsu sáng/tối/auto; morphology có thể tắt.
- [x] Đo chiều dài/rộng bằng minAreaRect, diện tích và chu vi contour.
- [x] Hiệu chỉnh bằng hai điểm tham chiếu (tọa độ hoặc chuột), hoặc tỷ lệ pixel/cm có sẵn.
- [x] ROI loại vật tham chiếu khỏi vùng phân đoạn; kiểm tra contour chạm viền.
- [x] Xuất ảnh kết quả, ảnh debug và CSV cho một/nhiều ảnh; ghi lỗi riêng từng ảnh.
- [x] Tạo data/demo.png tổng hợp và hướng dẫn thử nhanh.
- [x] Chạy 4 kiểm tra tự động: hình học xoay/quy đổi đơn vị, phân đoạn nền sáng/tối, mask rỗng/bị cắt, hiệu chỉnh không hợp lệ.
- [x] Chạy pipeline demo; xem trực quan ảnh results/demo.png/result.png.
- [x] Kiểm tra batch 2 ảnh hợp lệ + 1 ảnh hỏng: 2 kết quả, 1 dòng lỗi, mã thoát 1 đúng thiết kế.
- [x] Sửa lỗi mã hóa tiếng Việt khi stdout/stderr Windows chạy qua pipe.
- [x] Thêm app.py: kéo thả ảnh, chọn demo, chỉnh ROI, hiệu chỉnh bằng tọa độ đoạn tham chiếu hoặc pixel/cm.
- [x] Xem mặt nạ/ảnh trung gian, bốn số đo, tải PNG và CSV trực tiếp từ giao diện.
- [x] Không xuất số đo cm cho ảnh tải lên khi chưa nhập hiệu chỉnh; không giữ kết quả cũ khi cấu hình mới lỗi.
- [x] Thêm Streamlit vào requirements.txt và hướng dẫn chạy vào README.md.
- [x] Kiểm tra 7/7 bài đạt (4 bài xử lý ảnh + 3 bài giao diện AppTest).

## Đang làm dở
Không có thay đổi mã nguồn đang làm dở. Chưa có dữ liệu để đánh giá độ chính xác thực tế.

## Quyết định kỹ thuật và lý do
- Python, OpenCV, NumPy; không dùng deep learning.
- Otsu trên ảnh xám cho nền đơn giản; auto chọn cực tính từ viền ROI, cho phép ép dark/light khi cần.
- minAreaRect đơn giản, dễ trình bày và đo được cá xoay; định nghĩa chiều dài/rộng là cạnh hộp bao, có tính vây/đuôi trong silhouette.
- Hiệu chỉnh bán tự động bằng đoạn tham chiếu đã biết thay vì tự nhận dạng thước: dễ kiểm tra và tránh suy đoán sai tỷ lệ.
- ROI tùy chọn để tách thước/vật khác; chọn contour ngoài lớn nhất, từ chối contour chạm viền để tránh đo cá bị cắt.
- Dùng cm/cm² trong CSV; mm = cm × 10, mm² = cm² × 100.
- Lưu ảnh trung gian trước khi tìm contour để vẫn debug được khi phân đoạn thất bại.
- CSV UTF-8 BOM để đọc tiếng Việt trong Excel; mỗi lần chạy ghi lại kết quả cùng tên, có --output để giữ từng lần thử.
- Streamlit tái sử dụng các hàm xử lý hiện có, cập nhật theo thông số; PNG/CSV tải từ bộ nhớ, không ghi đè results của CLI.
- Giao diện web nhập tọa độ tham chiếu và hiển thị đường đánh dấu; chưa thêm thư viện chọn điểm trên ảnh để giữ phụ thuộc đơn giản.

## Kết quả kiểm tra phiên 2026-10-30
- Môi trường có sẵn: Python 3.14, OpenCV 4.13.0, NumPy 2.4.4.
- `python -m unittest discover -s tests -v`: 4/4 đạt.
- Demo: tham chiếu 200 px / 10 cm = 20 px/cm; dài 28.930304 cm,
  rộng 7.554401 cm, diện tích 162.01625 cm², chu vi 73.227774 cm.
- Đây là kết quả hình tổng hợp, không phải độ chính xác đã xác nhận trên cá thật.
- Batch kiểm tra nằm trong results/batch_validation/ (không đưa vào Git).
- `git diff --check`: không có lỗi whitespace.
- Streamlit 1.58.0 đã có trên máy. AppTest kiểm tra demo, ảnh tải lên (giả lập file uploader), thiếu hiệu chỉnh, tham chiếu trùng điểm, ROI nhỏ và ảnh hỏng.
- Khởi động máy chủ Streamlit trên 127.0.0.1:8501; có thể chạy lại bằng `python -m streamlit run app.py`.

## Vấn đề còn tồn đọng
- Rà soát theo câu hỏi độ tin cậy: 7/7 kiểm thử vẫn đạt nhưng chỉ xác nhận hình học tổng hợp và luồng giao diện, không chứng minh độ chính xác trên cá thật.
- Thử ảnh tổng hợp có cá dạng ellipse và vật tham chiếu tối lớn hơn: thuật toán chọn nhầm vật tham chiếu (bbox x=100, y=400, rộng=701, cao=61 pixel). Đây là giới hạn của chọn contour lớn nhất; cần cô lập ROI hoặc cải thiện chọn vùng trước khi coi phép đo là đáng tin.
- Chưa có ảnh cá thật, kích thước tham chiếu và số đo tay để tính sai số.
- Chưa kiểm tra tương tác bấm chuột/--show trực tiếp trên cửa sổ đồ họa; luồng tọa độ dòng lệnh đã chạy thành công.
- Otsu có thể sai khi nền phức tạp, bóng đổ, phản sáng hoặc nhiều vật thể; contour lớn nhất không bảo đảm là cá.
- Chưa xử lý méo ống kính/phối cảnh/độ cao thân cá; yêu cầu camera gần vuông góc, tham chiếu cùng mặt phẳng đo.
- Chiều dài hộp bao không thay thế chiều dài dọc thân cá cong; chu vi nhạy với nhiễu.

## Bước tiếp theo đề xuất
0. Trước khi thu thập ảnh hàng loạt: kiểm tra trên một vài ảnh cá thật có sẵn để đánh giá phân đoạn; thử đo một vật phẳng kích thước biết trước để kiểm tra hiệu chỉnh độc lập. Sau đó chỉ chụp 1–2 ảnh cá thí điểm có thước, đánh giá mask và số đo trước khi mở rộng bộ ảnh.
1. Nhận 3–5 ảnh cá Basa thật trên nền đơn giản, có đoạn tham chiếu biết độ dài và số đo tay dài/rộng.
2. Xác nhận quy ước đo chiều dài/rộng (có tính vây/đuôi hay không).
3. Chạy ảnh thật, xem mask và chỉnh ROI, cực tính, blur/morph; chỉ bổ sung phân đoạn màu nếu dữ liệu cho thấy cần.
4. So sánh với số đo tay, lập bảng sai số tuyệt đối/tương đối; lưu cấu hình thử nghiệm và ảnh minh họa để báo cáo.
5. Kiểm tra chọn tham chiếu bằng chuột trên máy người dùng; cập nhật PROGRESS.md cuối phiên.
6. Thử tải ảnh thật trên Streamlit; nếu nhập tọa độ bất tiện, cân nhắc bổ sung chọn điểm trực tiếp trên ảnh web.
