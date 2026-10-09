"""Kiểm tra toàn bộ luồng trên một mẫu nhỏ trước khi chạy dữ liệu đầy đủ."""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import sys
from pathlib import Path

from check_data import VIDEOS, check
from evaluate_practice import _load_eval_config
from run_kaggle_lab import run_logged, sequence_info


def filter_ground_truth(text: str, frame_count: int) -> str:
    """Giữ nhãn của đoạn đầu video_1 trong mẫu nhỏ, không đánh lại số frame.

    Args:
        text: Nội dung gt.txt gốc ở định dạng MOT với ít nhất chín cột.
        frame_count: Số frame đầu được lấy làm mẫu.

    Returns:
        Các dòng nhãn có số frame từ 1 đến frame_count.

    Raises:
        ValueError: Khi frame_count không dương hoặc nhãn không đúng định dạng.
    """
    if frame_count < 1:
        raise ValueError("Số frame mẫu phải lớn hơn 0.")
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        columns = line.split(",")
        if len(columns) < 9:
            raise ValueError("Nhãn video_1 cần ít nhất chín cột theo định dạng MOT.")
        try:
            values = [float(value.strip()) for value in columns[:9]]
            frame = int(values[0])
            if values[0] != frame or frame < 1:
                raise ValueError("Số frame không phải số nguyên dương.")
        except (ValueError, OverflowError) as exc:
            raise ValueError("Nhãn video_1 có giá trị không hợp lệ.") from exc
        if frame <= frame_count:
            rows.append(line)
    return "\n".join(rows) + ("\n" if rows else "")


def create_sample(source: Path, target: Path, frames: int = 20) -> dict[str, int]:
    """Tạo bản mẫu riêng từ ảnh đầu mỗi video và chỉ nhãn video_1.

    Args:
        source: Thư mục data_lab21 chứa năm video.
        target: Thư mục mới dùng cho mẫu, tách biệt dữ liệu gốc.
        frames: Số ảnh tối đa lấy từ đầu mỗi video.

    Returns:
        Số ảnh thực tế đã lấy theo video.

    Raises:
        ValueError: Khi số frame không hợp lệ hoặc ảnh không có tên MOT liên tiếp.
        FileExistsError: Khi thư mục mẫu đã tồn tại.
        FileNotFoundError: Khi thiếu ảnh, gt hoặc seqinfo của video_1.
    """
    if frames < 1:
        raise ValueError("Số frame mẫu phải lớn hơn 0.")
    if target.exists():
        raise FileExistsError("Thư mục mẫu đã tồn tại; chọn thư mục mới.")
    counts = {}
    # Kiểm tra cấu trúc trước khi sao chép để không đổi dữ liệu gốc.
    for video in VIDEOS:
        paths = sorted((source / video / "img1").glob("*.jpg"))[:frames]
        if not paths:
            raise FileNotFoundError(f"Thiếu ảnh {video}/img1.")
        if any(not path.stem.isdigit() or int(path.stem) != index
               for index, path in enumerate(paths, 1)):
            raise ValueError(f"{video}: ảnh cần đánh số liên tiếp từ 1 để khớp nhãn MOT.")
        counts[video] = len(paths)
    practice = source / "video_1"
    for relative in ("gt/gt.txt", "seqinfo.ini"):
        if not (practice / relative).is_file():
            raise FileNotFoundError(f"Thiếu video_1/{relative}.")
    cropped_gt = filter_ground_truth(
        (practice / "gt/gt.txt").read_text(encoding="utf-8"), counts["video_1"]
    )
    config = _load_eval_config(source)
    target.mkdir(parents=True)
    for video in VIDEOS:
        images = target / video / "img1"
        images.mkdir(parents=True)
        for path in sorted((source / video / "img1").glob("*.jpg"))[:frames]:
            shutil.copy2(path, images / path.name)
        original_info = source / video / "seqinfo.ini"
        if original_info.exists():
            info = configparser.ConfigParser()
            info.read(original_info, encoding="utf-8")
            info["Sequence"]["seqLength"] = str(counts[video])
            with (target / video / "seqinfo.ini").open("w", encoding="utf-8") as stream:
                info.write(stream)
    labels = target / "video_1/gt"
    labels.mkdir()
    (labels / "gt.txt").write_text(cropped_gt, encoding="utf-8")
    (target / "video_1/eval_config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    (target / "MAU_THU_KHONG_NOP.json").write_text(
        json.dumps({"purpose": "Mẫu kiểm tra luồng; không dùng số liệu này cho báo cáo chính",
                    "frames": counts}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return counts


def main() -> None:
    """Chạy luồng chính trên mẫu nhỏ và chỉ cho phép chạy lớn khi mẫu đã đạt.

    Raises:
        RuntimeError: Khi luồng mẫu không tạo đủ lượt, metric hoặc gói ZIP.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--trackeval-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--sample-root", type=Path, help="Thư mục ảnh mẫu tạm, tách khỏi kết quả")
    parser.add_argument("--cache-root", type=Path, help="Thư mục cache mẫu tạm")
    parser.add_argument("--frames", default=20, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not check(args.lab_data_root):
        raise RuntimeError("Dữ liệu gốc chưa đủ để kiểm tra mẫu.")
    for video in VIDEOS:
        sequence_info(args.lab_data_root, video)
    work = args.work_root.resolve()
    sample = args.sample_root.resolve() if args.sample_root else work / "data_mau"
    counts = create_sample(args.lab_data_root.resolve(), sample, args.frames)
    work.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    command = [
        sys.executable, str(scripts / "run_kaggle_lab.py"),
        "--lab-data-root", str(sample),
        "--trackeval-root", str(args.trackeval_root.resolve()),
        "--out", str(work / "ket_qua_mau"), "--device", args.device,
        "--student", "Mẫu kiểm tra luồng", "--student-id", "SMOKE",
    ]
    if args.cache_root:
        command.extend(["--cache-root", str(args.cache_root.resolve())])
    run_logged(command, work / "kiem_tra_mau.log")
    manifest = json.loads((work / "ket_qua_mau/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or len(manifest["records"]) != 15:
        raise RuntimeError("Mẫu chưa chạy đủ 15 lượt.")
    if not (work / "ket_qua_SMOKE.zip").is_file():
        raise RuntimeError("Mẫu chưa tạo được ZIP.")
    (work / "DAT_KIEM_TRA.json").write_text(
        json.dumps({"status": "passed", "frames": counts, "runs": 15,
                    "warning": "Chỉ kiểm tra luồng; không phải kết quả nộp bài"},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nĐẠT KIỂM TRA MẪU: tracking, cache, preview, đánh giá video_1 và đóng ZIP.")
    print("Có thể chạy dữ liệu đầy đủ. Kiểm tra mẫu không bảo đảm mọi frame còn lại đều hợp lệ.")


if __name__ == "__main__":
    main()
