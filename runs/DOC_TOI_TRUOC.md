# Bài nộp lab tracking — Lưu Quang Khải, MSSV 2A202602599

Báo cáo: submission_template/BAO_CAO_mau.md.

Năm file nộp: runs/nop_bai/video_1.txt … video_5.txt.

Video_1 và video_4 dùng BoT-SORT conf=0.30; video_2 và video_5 dùng BoT-SORT conf=0.15; video_3 dùng ByteTrack conf=0.30. Tất cả iou=0.50. File nộp lấy nguyên từ các lượt đã chạy đủ frame, không chạy lại mô hình hoặc sửa tọa độ/ID.

metrics/: summary và bảng chi tiết chính thức của video_1 cho cả ba cấu hình. ket_qua_thu_nghiem.csv: dữ kiện 15 lượt. cau_hinh_nop_va_kiem_tra.json: cấu hình cuối và hash kiểm tra.

minh_chung/: ảnh mốc của cấu hình chọn và ảnh ghép đối chiếu ba cấu hình trên cả năm video. Nhận xét báo cáo giới hạn ở các ảnh mẫu đã xem. ZIP Kaggle gốc giữ nguyên và có các đoạn video để kiểm tra thêm; không khẳng định đã xem liên tục mọi frame.

Gói này không chứa dữ liệu gốc, nhãn hay trọng số. Không commit các video preview hoặc ZIP Kaggle có video vào Git.
