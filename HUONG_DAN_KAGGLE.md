# Chạy lab trên Kaggle bằng Save & Run All

Sinh viên: **Lưu Quang Khải** · MSSV: **2A202602599**.

## 1. Chuẩn bị một lần trên máy local

Tại thư mục gốc repo, chạy:

```powershell
python scripts/build_kaggle_upload.py
```

File `runs/kaggle_upload.zip` chỉ chứa mã nguồn, notebook, test và hướng dẫn.
Tạo Dataset **Private** trên Kaggle, gắn `kaggle_upload.zip` và dữ liệu lab do lớp phát
(có thể dùng ZIP gốc hoặc thư mục đã giải nén). Giữ cả Notebook ở chế độ Private.
Chỉ tải dữ liệu lab lên dịch vụ bên ngoài khi quy định của lớp cho phép.

## 2. Tạo Notebook và chạy

1. Import `kaggle_run_all.ipynb` từ repo vào một Notebook Kaggle.
2. Add Input: gắn Dataset chứa mã nguồn và Dataset chứa dữ liệu nếu tách riêng.
3. Settings: bật **GPU** và **Internet**. Luồng dùng một GPU `cuda:0`.
4. Chọn **Save Version → Save & Run All** để chạy từ đầu trong phiên mới.

Không cần sửa đường dẫn nếu chỉ có một gói mã nguồn và một bộ dữ liệu lab hợp lệ
trong Input. Nếu có nhiều bản, điền `REPO_HINT` / `LAB_DATA_HINT` ở ô đầu tiên bằng
đường dẫn thư mục đã giải nén trong `/kaggle/input`. Với ZIP, bỏ bản không dùng khỏi Input.
Notebook tự giải nén ZIP có cấu trúc lab/repo, kiểm tra dữ liệu, cài môi trường riêng,
kiểm tra GPU, chạy toàn bộ luồng trên tối đa **20 frame đầu mỗi video**,
rồi mới chạy dữ liệu đầy đủ khi mẫu đã đạt.

Cell **Kiểm tra toàn bộ luồng trên mẫu nhỏ** chạy cả ba cấu hình trên năm video,
kiểm tra cache detection, xuất preview, trích ảnh/video bằng ffmpeg, chấm `video_1`
và đóng ZIP. Nó tạo bản mẫu riêng trong `/kaggle/temp`, rút `seqLength` và nhãn
`video_1` theo số ảnh mẫu; không thay đổi dữ liệu gốc hoặc tạo nhãn cho bốn video kia.
Nếu lỗi, cell dừng trước bước chạy lớn. Khi đạt sẽ in **ĐẠT KIỂM TRA MẪU**.
Điểm của mẫu chỉ dùng kiểm tra luồng, không đưa vào báo cáo chính. Mẫu nhỏ không
thể bảo đảm mọi ảnh trong phần dữ liệu còn lại đều đọc được.

Kết quả mẫu được lưu tại `/kaggle/working/kiem_tra_mau`, còn ảnh mẫu và cache ở
`/kaggle/temp/tracking_lab` để không xuất lại dữ liệu gốc. Log cài đặt được ghi
vào `/kaggle/working/logs_cai_dat`, log bao quát hai bước chạy vào
`/kaggle/working/logs_chay`. Các log được ghi liên tục trong khi chạy.

Thư mục gốc có thể tên `data_lab21`; notebook tìm thư mục chứa `video_1`…`video_5`,
không phụ thuộc tên thư mục gốc. Notebook kiểm tra **có file ảnh thật và nhãn**,
không chọn thư mục chỉ có các thư mục con rỗng. Cấu trúc tương thích:

```text
data_lab21/
  video_1/
    img1/             # Ảnh JPG đánh số liên tiếp từ 1
    gt/gt.txt         # Nhãn để đánh giá video_1
    seqinfo.ini       # FPS và tổng số frame
    det/det.txt       # Không dùng; detection do YOLO cố định tạo ra
  video_2/img1/
  video_3/img1/
  video_4/img1/
  video_5/img1/
```

`eval_config.json` là tùy chọn. Nếu thiếu, dùng tên thư mục đánh giá `LAB/train`
với nhãn và `seqinfo.ini` của `video_1`; không suy ra hoặc thay nhãn. Nếu file có sẵn,
giữ cấu hình đánh giá được phát cùng dữ liệu.

Nếu báo `0 ảnh` và thiếu `gt.txt`, kiểm tra Input đang gắn đúng phiên bản Dataset
có dữ liệu đầy đủ. Xem đường dẫn và danh sách file trong thông báo chẩn đoán;
gói ZIP cần được giải nén. Chọn `LAB_DATA_HINT` là thư mục ngay phía trên `video_1`,
không chọn thư mục cha cao hơn. File ảnh hiện được đọc bằng hậu tố `.jpg` chữ thường.
`check_data.py` trả mã lỗi 1 nếu thiếu dữ liệu, để notebook dừng trước tải trọng số và chạy mẫu.

Lần cài đầu cần Internet để tải Python, thư viện, trọng số và TrackEval. Notebook tự
cài **Python 3.11 riêng bằng uv**, kể cả khi kernel Kaggle dùng Python 3.13. Không cần
đổi kernel hoặc image. Kernel vẫn có thể hiển thị Python 3.13; các script lab chạy
bằng `/kaggle/temp/tracking_lab/venv_py311/bin/python`.

Môi trường riêng cài PyTorch `2.6.0`, torchvision `0.21.0` từ index CUDA 12.4,
dùng BoxMOT `10.0.42` và NumPy `1.26.4`; không dùng resolver
mặc định của BoxMOT vì nó ghim NumPy `1.23.1`, không phù hợp Python mới.
Không lấy các thư viện nhị phân của Python 3.13 vào môi trường Python 3.11.
Lần đầu sẽ tải thêm PyTorch/CUDA nên phần cài đặt có thể mất nhiều phút.
Notebook kiểm tra `nvidia-smi` trước cài đặt và thử một phép tính CUDA trước chạy mẫu.
Đặt `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` cho checkpoint Re-ID cố định của lab vì
BoxMOT cũ gọi `torch.load` theo hành vi mặc định cũ. Không thay mô hình hoặc trọng số.

## 3. Notebook thực hiện những gì?

Chạy **đủ frame cho tất cả 15 lượt**, gồm ba cấu hình cho mỗi video:

| Tên | Tracker | conf | iou |
|---|---|---|---|
| bytetrack_c030_i050 | ByteTrack | 0.30 | 0.50 |
| botsort_c030_i050 | BoT-SORT | 0.30 | 0.50 |
| botsort_c015_i050 | BoT-SORT | 0.15 | 0.50 |

Hai cấu hình đầu giữ nguyên detector để so sánh tracker. Hai cấu hình BoT-SORT
chỉ đổi `conf`. Các lượt cùng ảnh, `conf` và `iou` dùng chung cache detection:
YOLO chạy hai lần/video thay vì ba lần. Re-ID và tracking vẫn chạy riêng.
Cache ghi đủ frame mới được công nhận; metadata kiểm tra đường dẫn, danh sách ảnh,
kích thước/thời điểm sửa file và cấu hình detector. Không dùng cache từ dữ liệu đã thay đổi.
Cache không kiểm tra hash nội dung từng ảnh; tránh sửa dữ liệu trong khi chạy.

Giữ cố định `yolo26n.pt`, ảnh 640, lớp người và Re-ID `osnet_x0_25_msmt17.pt`.
Không giới hạn frame trong thí nghiệm chính. Kiểm tra số frame thực tế với số ảnh,
và với `seqLength` nếu có `seqinfo.ini`. Ảnh hỏng khiến luồng dừng thay vì bỏ qua.

Chỉ `video_1` được chấm HOTA/MOTA/IDF1. Chọn kết quả có HOTA cao nhất trong
ba cấu hình đã thử; nếu hòa, ưu tiên IDF1 rồi MOTA. Đây là lựa chọn trên video luyện,
không phải bằng chứng cấu hình tối ưu cho mọi cảnh.

`video_2`–`video_5` **tạm chọn** BoT-SORT `conf=0.3`, `iou=0.5`. Vì không có nhãn,
cần xem video để xác nhận hoặc chọn cấu hình khác. Mọi cấu hình đều đã có file TXT
đủ frame, nên đổi lựa chọn sau khi xem không cần chạy lại. Không suy chất lượng
tracking chỉ từ số lượng ID hoặc hộp.

Thời gian phụ thuộc GPU, số ảnh và mật độ người. Luồng này chạy một phiên từ đầu
đến cuối, không có nghĩa là xử lý tức thì. Log in tiến độ mỗi 100 frame.
Thời gian/FPS trong từng lượt gồm nạp mô hình và có thể dùng cache; không dùng
cột này để kết luận tốc độ tracker độc lập khi điều kiện cache khác nhau.

## 4. Tải kết quả và gửi lại

Khi chạy thành công, vào Output của phiên đã lưu, tải:

```text
/kaggle/working/ket_qua_2A202602599.zip
```

ZIP chứa:

- `nop_bai/video_1.txt` … `video_5.txt`: kết quả đã chọn/tạm chọn.
- `thu_nghiem/`: 15 file TXT đủ frame và thông tin từng lượt, dùng khi đổi lựa chọn.
- `metrics/`: summary và bảng chi tiết chính thức của `video_1`.
- `ket_qua_thu_nghiem.csv`, `manifest.json`: cấu hình, số frame, thời gian, môi trường và lý do chọn.
- `logs/`: log tracking/đánh giá và phiên bản thư viện.
- `minh_chung/`: ba ảnh và ba đoạn video khoảng 8 giây tại 10%, 50%, 90% của mỗi lượt.
- `BAO_CAO_nhap.md`: tên/MSSV, cấu hình và metric thật; chưa tự tạo quan sát.

Các kết quả cần giữ đều nằm trong `/kaggle/working`:

```text
/kaggle/working/
  ket_qua_2A202602599.zip       # Gói kết quả chính để gửi lại
  ket_qua_lab/                # Toàn bộ thí nghiệm chính, preview, báo cáo và log
  kiem_tra_mau/               # Kết quả mẫu và ket_qua_SMOKE.zip, không dùng nộp bài
  logs_cai_dat/               # Log tải Python, PyTorch và cài thư viện
  logs_chay/                  # Log bao quát kiểm tra mẫu và chạy toàn bộ
```

`/kaggle/temp` chỉ giữ mã nguồn đang chạy, môi trường Python, trọng số, cache và
dữ liệu giải nén/mẫu. Những file này có thể tạo lại; không cần lưu cùng kết quả.
Sau Save & Run All, tải file từ Output của **phiên đã lưu**, không dựa vào thư mục
tạm trong phiên tương tác. Giữ kết quả trong `/kaggle/working` không thay thế việc
lưu phiên chạy với Output.

Preview **đầy đủ** của mọi cấu hình vẫn nằm tại
`/kaggle/working/ket_qua_lab/thu_nghiem/<cấu_hình>/<video>_preview.mp4`;
tải riêng nếu cần kiểm tra đoạn khác. ZIP không chứa ảnh gốc, nhãn hoặc trọng số.
Preview giới hạn chiều rộng 1280 px để tiết kiệm dung lượng; ảnh đưa vào detector
và tọa độ trong TXT vẫn giữ nguyên quy định lab.
Cache và môi trường cài đặt nằm trong `/kaggle/temp` để không chiếm phần Output lưu lại.

Gửi ZIP kết quả lại trong cuộc trò chuyện để xem minh chứng, chốt cấu hình cho
bốn video không nhãn và hoàn thiện báo cáo theo mẫu. Các đoạn trích là mẫu quan sát,
không bao phủ tất cả tình huống; có thể cần preview đầy đủ để xác minh một nhận xét.
Nếu cần đổi lựa chọn, sao chép TXT tương ứng từ `thu_nghiem` vào `nop_bai`, rồi
cập nhật báo cáo cho khớp. Không cần chạy lại mô hình.

Nếu có lỗi, notebook dừng và không tuyên bố hoàn thành. Lỗi trong giai đoạn thí nghiệm
được lưu thành `LOI.json` cùng gói kết quả một phần; gửi log để xử lý, không nộp gói lỗi.

## 5. Chạy từ terminal khi cần

Sau khi cài các phụ thuộc, có thể dùng cùng luồng ngoài notebook:

```bash
python scripts/run_kaggle_lab.py --lab-data-root /duong/dan/lab_data --trackeval-root /duong/dan/TrackEval --out runs/kaggle --device cuda:0
```

Chọn thư mục `--out` trống/mới mỗi lần để không trộn kết quả cũ. Trong notebook,
thư mục đầu ra được tạo riêng trong phiên Save & Run All mới.

Tài liệu nền tảng: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).
Quản lý Python: [uv](https://docs.astral.sh/uv/guides/install-python/).
Cặp PyTorch/torchvision và index CUDA: [PyTorch](https://pytorch.org/get-started/previous-versions/).
