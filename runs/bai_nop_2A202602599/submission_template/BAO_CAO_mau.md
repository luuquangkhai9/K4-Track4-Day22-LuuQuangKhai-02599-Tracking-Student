# Báo cáo lab: chọn tracker cho 5 video

**Sinh viên:** Lưu Quang Khải · **MSSV:** 2A202602599

Detector cố định: `yolo26n.pt`, ảnh đầu vào 640 px, chỉ lớp người; Re-ID `osnet_x0_25_msmt17.pt`. Không thay các thành phần này trong thí nghiệm.

Thực nghiệm chạy trên Kaggle với GPU Tesla T4, Python 3.11.16, PyTorch 2.6.0+cu124, Ultralytics 8.4.174 và BoxMOT 10.0.42. Mỗi video chạy đủ frame với ba cấu hình: ByteTrack (`conf=0.3`, `iou=0.5`), BoT-SORT (`conf=0.3`, `iou=0.5`) và BoT-SORT (`conf=0.15`, `iou=0.5`). Hai lượt đầu chỉ đổi tracker; hai lượt BoT-SORT chỉ đổi `conf`. Chưa quét thêm `iou` trong đợt thực nghiệm này.

Các lượt cùng cấu hình detector dùng lại detection đã lưu; tracker vẫn khởi tạo riêng và xử lý từ đầu. Số frame lần lượt là 600, 1.050, 837, 900 và 750, khớp ảnh gốc: tổng cộng 4.137 frame dữ liệu và 12.411 lượt xử lý frame qua ba cấu hình. Tổng thời gian tracking ghi trong 15 lượt là khoảng 24 phút 44 giây, chưa gồm cài đặt, chấm metric và trích minh chứng. Do một số lượt dùng cache còn một số lượt chạy YOLO, FPS ghi trong log không được dùng để kết luận tốc độ tracker độc lập.

Số liệu lấy từ `metrics/*/pedestrian_summary.txt`, đối chiếu với manifest và bảng CSV trong `ket_qua_2A202602599.zip`. Nhận xét định tính dựa trên các ảnh minh chứng tại ba mốc khoảng 10%, 50%, 90% của mỗi video. Đây là quan sát theo mẫu, không phải kiểm tra liên tục mọi frame; không suy số lần đổi ID hoặc chất lượng toàn video từ số lượng ID trong file.

## 1. Cấu hình đã chọn

Các lựa chọn cho `video_2`–`video_5` được chốt theo minh chứng hiện có. Các nhận xét về mức độ bao phủ người là quan sát bằng mắt, không phải recall tính từ nhãn.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.30 | 0.50 | Frame 301: có hộp cho người nhỏ gần cửa hàng bên trái và người mặc áo tối đứng cạnh người lớn tuổi ở bên phải, trong khi ByteTrack thiếu các hộp này. Người rất xa vẫn thường không có hộp. | ByteTrack 0.30/0.50 có HOTA thấp hơn; BoT-SORT 0.15/0.50 có MOTA và IDF1 cao hơn một chút nhưng HOTA thấp hơn cấu hình chọn. |
| video_2 (phố đêm, tĩnh, rất đông) | botsort | 0.15 | 0.50 | Frame 526: bổ sung hộp cho người áo sơ mi ở nửa trái phía trên và người áo sáng/túi tối bên phải. Frame 946: có thêm hộp cho người quần sáng ở phía trái và người ở xa, trong khi 0.30 bỏ sót. Nhiều người ở góc trên trái vẫn chưa có hộp. | BoT-SORT 0.30/0.50 và ByteTrack 0.30/0.50 bỏ sót nhiều người rõ trong các ảnh đối chiếu. |
| video_3 (camera di động, ảnh nhỏ) | bytetrack | 0.30 | 0.50 | Frame 419: cả ba cấu hình bám bốn người chính tương tự nhau. Frame 754: vẫn có hộp cho người đi ngược chiều bên trái và nhóm đi cùng chiều ở giữa; chưa thấy lợi ích rõ của hạ conf ở các ảnh mẫu. | Hai cấu hình BoT-SORT chưa cho khác biệt rõ về người được bám tại ba mốc; chọn phương án không cần Re-ID khi chưa có bằng chứng lợi ích đủ rõ trên ảnh nhỏ. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.30 | 0.50 | Frame 91 và 451: có hộp cho nhóm áo đỏ, áo chấm bi và áo trắng giữa hành lang. Frame 811: nhóm người đi ngược chiều cũng được bám, nhưng người sát mép ảnh bị cắt thân. Người áo đỏ mang ID 2 ở hai mốc đầu, ID 89 ở mốc cuối; liên kết xuyên suốt còn hạn chế. | BoT-SORT 0.15/0.50 chưa cải thiện rõ nhóm cận cảnh ở ba mốc. ByteTrack 0.30/0.50 cho hộp tương tự; giữ BoT-SORT để sử dụng thêm ngoại hình trong cảnh nhiều người giao nhau, chưa khẳng định nó tốt hơn toàn video. |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.15 | 0.50 | Frame 76: có thêm hộp cho người trong bóng tối ở vỉa hè trái. Frame 376: bám nhóm đứng chờ ở góc phải; BoT-SORT còn có hộp cho người ở xa sát mép phải mà ByteTrack không có. Frame 676: cả ba cấu hình đều thiếu hộp cho người đang sang đường mang túi đỏ. | ByteTrack 0.30/0.50 và BoT-SORT 0.30/0.50 bao phủ ít người hơn ở vỉa hè trái tại frame 76; hạ conf vẫn không xử lý được mọi trường hợp bỏ sót. |

## 2. Số liệu video_1

Kết quả thực tế do `scripts/evaluate_practice.py` xuất, theo thang điểm 0–100:

| Cấu hình | HOTA | MOTA | IDF1 |
|---|---:|---:|---:|
| ByteTrack, conf=0.30, iou=0.50 | 26.912 | 17.292 | 25.713 |
| **BoT-SORT, conf=0.30, iou=0.50 — chọn** | **29.460** | **19.811** | **29.354** |
| BoT-SORT, conf=0.15, iou=0.50 | 29.343 | 20.731 | 29.561 |

Dòng kết quả của cấu hình nộp, chép từ bảng TrackEval:

```
HOTA:     video_1  HOTA=29.46  DetA=18.095  AssA=48.223
CLEAR:    video_1  MOTA=19.811  CLR_Re=21.759  CLR_Pr=92.306
                  CLR_TP=4043  CLR_FN=14538  CLR_FP=337  IDSW=25
Identity: video_1  IDF1=29.354  IDR=18.137  IDP=76.941
```

BoT-SORT 0.30 tăng HOTA 2.548 điểm, MOTA 2.519 điểm và IDF1 3.641 điểm so với ByteTrack 0.30. Tuy nhiên, AssA chỉ tăng từ 48.130 lên 48.223; mức tăng HOTA đi kèm mức tăng DetA từ 15.068 lên 18.095, nên không quy toàn bộ cải thiện cho Re-ID. ByteTrack có 12 lần đổi ID theo CLEAR, thấp hơn 25 lần của BoT-SORT 0.30, nhưng nó cũng có recall thấp hơn và bỏ sót nhiều người hơn. Vì vậy, chỉ nhìn IDSW thấp không đủ để chọn tracker.

Khi hạ conf của BoT-SORT từ 0.30 xuống 0.15, số bỏ sót giảm từ 14.538 xuống 14.197, nhưng phát hiện giả tăng từ 337 lên 505 và AssA giảm từ 48.223 xuống 45.113. MOTA tăng 0.920 điểm và IDF1 tăng 0.207 điểm, trong khi HOTA giảm 0.117 điểm. Chọn 0.30 theo tiêu chí HOTA đặt trước; khoảng cách HOTA nhỏ, không có kiểm định để khẳng định ưu thế ổn định của 0.30.

`video_2` đến `video_5` không có nhãn dùng đánh giá trong gói lab. Không tính hoặc điền HOTA/MOTA/IDF1 cho các video đó.

## 3. Phân tích

### video_1 — cân bằng phát hiện và liên kết

Ở quảng trường ban ngày, người gần camera có hộp khá rõ, còn nhiều người nhỏ ở nền xa không có hộp. Frame 301 cho thấy BoT-SORT 0.30 bổ sung một số track mà ByteTrack không xuất ra dù hai lượt dùng cùng detection; khác biệt này nằm ở xử lý của tracker. HOTA 29.460 là cao nhất trong ba cấu hình, nhưng CLR_Re 21.759 cho thấy hệ thống vẫn bỏ sót phần lớn nhãn dùng chấm. Điểm HOTA tăng chủ yếu đi cùng cải thiện phát hiện/độ bao phủ track, trong khi AssA gần như ngang ByteTrack. Vì vậy, lựa chọn BoT-SORT 0.30 là lựa chọn tốt nhất theo HOTA trong các cấu hình đã thử, không phải kết luận hệ thống đã tracking tốt tuyệt đối.

### video_2 — giảm ngưỡng để bớt mất người trong cảnh tối

Camera nhìn từ trên cao, cảnh tối và đông làm nhiều người chỉ chiếm một vùng nhỏ hoặc bị lẫn vào nền. Tại frame 526 và 946, BoT-SORT 0.15 giữ thêm hộp trên một số người nhìn thấy được mà cấu hình 0.30 không có hộp. Vì những hộp bổ sung này trùng với người thực trong ảnh mẫu, chọn ngưỡng thấp hơn có căn cứ quan sát cho cảnh này. Tuy vậy, nhóm đông ở góc trên trái vẫn bị bỏ sót nhiều, và không có nhãn để xác nhận mức cải thiện trên toàn bộ video. Việc dùng Re-ID nhằm bổ sung thông tin ngoại hình khi người giao nhau; ba ảnh mẫu chưa đủ để đo khả năng giữ ID liên tục.

### video_3 — chưa thấy lợi ích rõ của Re-ID trên ảnh nhỏ

Ảnh có độ phân giải thấp, người ở xa nhỏ và các mốc cho thấy camera tiến dọc vỉa hè. Ở frame 419, cả ByteTrack và hai cấu hình BoT-SORT có hộp cho bốn người chính; frame 754 cũng cho kết quả bao phủ nhóm tiền cảnh gần nhau. Hạ conf không tạo lợi ích rõ tại các mốc đã xem, nên chọn ByteTrack 0.30 làm phương án đơn giản không cần đặc trưng ngoại hình. Đây là quyết định theo mẫu và chi phí xử lý, chưa phải bằng chứng ByteTrack giữ ID tốt hơn BoT-SORT trên toàn chuỗi. Camera di chuyển và che khuất vẫn là các yếu tố có thể làm liên kết theo chuyển động gặp khó khăn.

### video_4 — có Re-ID nhưng vẫn mất liên kết dài hạn

Camera tiến trong hành lang có kính và sàn phản chiếu, đồng thời xuất hiện người đi cùng và ngược chiều. Tại frame 91 và 451, BoT-SORT 0.30 có hộp cho nhóm áo đỏ, áo chấm bi và áo trắng giữa ảnh, tương tự các cấu hình còn lại. Người áo đỏ/quần be cùng nhóm mang ID 2 ở hai mốc đầu và ID 89 ở mốc cuối, cho thấy lựa chọn có Re-ID vẫn chưa duy trì được danh tính xuyên suốt các mốc quan sát. Giữ conf 0.30 vì hạ xuống 0.15 chưa tạo khác biệt rõ ở nhóm cận cảnh trong ba ảnh, và sử dụng thêm ngoại hình là một lựa chọn hợp lý để thử trong cảnh người giao nhau. Không kết luận cấu hình này xử lý phản chiếu hoặc đổi ID tốt hơn ByteTrack khi chưa có nhãn và chưa xem liên tục mọi frame.

### video_5 — cải thiện vùng tối, còn bỏ sót khi sang đường

Các mốc cho thấy góc nhìn từ xe thay đổi mạnh khi đi qua giao lộ, trong khi người trên vỉa hè có kích thước nhỏ. Frame 76 cho thấy BoT-SORT 0.15 thêm hộp cho người trong bóng tối bên trái, nên chọn cấu hình này thay vì giữ lựa chọn tạm 0.30. Ở frame 376, nhóm người chờ bên phải được bám nhưng nhiều người phía xa vẫn không có hộp. Đặc biệt, người mang túi đỏ đang sang đường tại frame 676 không có hộp ở cả ba cấu hình, nên hạ conf và dùng Re-ID không khắc phục được mọi trường hợp bỏ sót. Lựa chọn cuối ưu tiên độ bao phủ thấy được ở ảnh mẫu, không khẳng định khả năng giữ ID xuyên suốt toàn video.

## 4. Nếu có thêm thời gian

Sẽ xem liên tục các đoạn che khuất và người sang đường, nhất là quanh frame 676 của video_5, đồng thời đối chiếu detection trước tracker để xác định bỏ sót bắt đầu ở bước nào. Tiếp theo có thể quét conf quanh 0.15–0.30 và kiểm tra ảnh hưởng thực tế của iou, vẫn giữ detector, kích thước ảnh và Re-ID cố định theo quy định lab.

## 5. File nộp và minh chứng

| File | Số frame đã xử lý | Cấu hình tương ứng |
|---|---:|---|
| runs/nop_bai/video_1.txt | 600 | botsort_c030_i050 |
| runs/nop_bai/video_2.txt | 1.050 | botsort_c015_i050 |
| runs/nop_bai/video_3.txt | 837 | bytetrack_c030_i050 |
| runs/nop_bai/video_4.txt | 900 | botsort_c030_i050 |
| runs/nop_bai/video_5.txt | 750 | botsort_c015_i050 |

Đã đối chiếu số frame với dữ liệu gốc và kiểm tra 15 file thử có 10 cột MOT, số frame/ID nguyên, ID dương, tọa độ hữu hạn, kích thước hộp dương và không trùng ID trong cùng frame. File TXT chứa một dòng cho mỗi hộp tracking, nên số dòng không bằng số frame. Các kết quả đã chạy đủ frame được lấy trực tiếp từ thư mục thí nghiệm tương ứng, không sửa ID hoặc tọa độ để tạo bài nộp.

Gói bài nộp kèm `minh_chung/doi_chieu_anh/video_N_so_sanh.jpg` cho cả năm video, ảnh mốc của cấu hình chọn và các summary chính thức của video_1. ZIP kết quả Kaggle gốc còn giữ các đoạn video trích để có thể kiểm tra thêm; kết luận định tính trong báo cáo này giới hạn ở các ảnh mốc đã đối chiếu.
