# Bản nháp báo cáo lab tracking

**Sinh viên:** Lưu Quang Khải · **MSSV:** 2A202602599

Detector cố định: `yolo26n.pt`, ảnh 640 px, lớp người; Re-ID `osnet_x0_25_msmt17.pt`.

Bản nháp tự điền cấu hình và số liệu thực tế. Cần xem minh chứng để hoàn thiện quan sát và xác nhận lựa chọn cho video_2–video_5 trước khi nộp.

## 1. Cấu hình

| Video | Cấu hình | Tracker | conf | iou | Cách chọn |
|---|---|---|---|---|---|
| video_1 | botsort_c030_i050 | botsort | 0.3 | 0.5 | HOTA cao nhất trong ba cấu hình; hòa điểm xét IDF1 rồi MOTA |
| video_2 | botsort_c030_i050 | botsort | 0.3 | 0.5 | Tạm chọn; cần xem minh chứng để xác nhận hoặc đổi cấu hình |
| video_3 | botsort_c030_i050 | botsort | 0.3 | 0.5 | Tạm chọn; cần xem minh chứng để xác nhận hoặc đổi cấu hình |
| video_4 | botsort_c030_i050 | botsort | 0.3 | 0.5 | Tạm chọn; cần xem minh chứng để xác nhận hoặc đổi cấu hình |
| video_5 | botsort_c030_i050 | botsort | 0.3 | 0.5 | Tạm chọn; cần xem minh chứng để xác nhận hoặc đổi cấu hình |

## 2. Số liệu video_1

Điểm theo thang 0–100 của TrackEval; MOTA có thể âm.

| Cấu hình | HOTA | MOTA | IDF1 |
|---|---|---|---|
| bytetrack_c030_i050 | 26.912 | 17.292 | 25.713 |
| botsort_c030_i050 | 29.460 | 19.811 | 29.354 |
| botsort_c015_i050 | 29.343 | 20.731 | 29.561 |

video_2–video_5 không có nhãn; không tính metric cho các video này.

## 3. Quan sát và cấu hình đã loại

### video_1

- Quan sát: [cần xem minh_chung và ghi frame/đoạn cụ thể].
- Cấu hình đã thử nhưng loại: [điền sau khi so sánh].

### video_2

- Quan sát: [cần xem minh_chung và ghi frame/đoạn cụ thể].
- Cấu hình đã thử nhưng loại: [điền sau khi so sánh].

### video_3

- Quan sát: [cần xem minh_chung và ghi frame/đoạn cụ thể].
- Cấu hình đã thử nhưng loại: [điền sau khi so sánh].

### video_4

- Quan sát: [cần xem minh_chung và ghi frame/đoạn cụ thể].
- Cấu hình đã thử nhưng loại: [điền sau khi so sánh].

### video_5

- Quan sát: [cần xem minh_chung và ghi frame/đoạn cụ thể].
- Cấu hình đã thử nhưng loại: [điền sau khi so sánh].

## 4. Phân tích

[Viết 3–5 câu cho ít nhất hai video, liên hệ cảnh với lỗi tracking đã thấy.]

## 5. Nếu có thêm thời gian

[Điền sau khi xem kết quả.]
