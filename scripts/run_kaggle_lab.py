"""Chạy thí nghiệm đủ frame, chấm video_1 và đóng gói minh chứng cho Kaggle."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from check_data import VIDEOS, check
from evaluate_practice import _load_eval_config
from tracking_artifacts import best_practice_case, parse_summary

DEFAULT_CASES = [
    {"name": "bytetrack_c030_i050", "tracker": "bytetrack", "conf": 0.3, "iou": 0.5},
    {"name": "botsort_c030_i050", "tracker": "botsort", "conf": 0.3, "iou": 0.5},
    {"name": "botsort_c015_i050", "tracker": "botsort", "conf": 0.15, "iou": 0.5},
]


def run_logged(command: list[str], log_path: Path, env_overrides: dict | None = None) -> None:
    """Chạy lệnh, hiển thị tiến độ và lưu nguyên văn log để kiểm tra sau.

    Args:
        command: Các đối số lệnh, không dùng shell.
        log_path: File log UTF-8.
        env_overrides: Biến môi trường riêng cho lệnh, ví dụ đường dẫn Python của uv.

    Raises:
        subprocess.CalledProcessError: Khi lệnh thất bại.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if env_overrides:
        env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(command, ensure_ascii=False) + "\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", env=env) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def sequence_info(root: Path, video: str) -> tuple[int, float]:
    """Đếm ảnh và lấy FPS của video từ seqinfo nếu có.

    Args:
        root: Thư mục lab_data.
        video: Tên video.

    Returns:
        Số ảnh và FPS để hiển thị preview; mặc định 20 khi thiếu seqinfo.

    Raises:
        ValueError: Khi số ảnh không khớp seqLength hoặc FPS không hợp lệ.
    """
    count = len(list((root / video / "img1").glob("*.jpg")))
    info = root / video / "seqinfo.ini"
    fps = 20.0
    if info.exists():
        parser = configparser.ConfigParser()
        parser.read(info, encoding="utf-8")
        seq = parser["Sequence"]
        if seq.getint("seqLength") != count:
            raise ValueError(f"{video}: số ảnh không khớp seqLength.")
        fps = seq.getfloat("frameRate")
    if count < 1 or not 0 < fps < 1000:
        raise ValueError(f"{video}: số ảnh hoặc FPS không hợp lệ.")
    return count, fps


def make_evidence(preview: Path, output: Path, count: int, fps: float) -> None:
    """Trích ảnh và ba đoạn video tại đầu, giữa, cuối để dễ gửi lại xem.

    Args:
        preview: Video đầy đủ đã vẽ ID.
        output: Thư mục chứa minh chứng cho cấu hình này.
        count: Số frame đã xử lý.
        fps: FPS của preview.

    Raises:
        RuntimeError: Khi không đọc được ảnh minh chứng.
        subprocess.CalledProcessError: Khi ffmpeg không xuất được đoạn video.
    """
    import cv2

    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(preview))
    try:
        for number, fraction in enumerate((0.1, 0.5, 0.9), 1):
            index = min(count - 1, int(count * fraction))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Không đọc được frame {index + 1} của {preview}")
            width = min(1280, image.shape[1])
            height = round(image.shape[0] * width / image.shape[1])
            image = cv2.resize(image, (width, height))
            cv2.putText(image, f"Frame {index + 1}/{count}", (12, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            target = output / f"mau_{number}_frame_{index + 1}.jpg"
            if not cv2.imwrite(str(target), image):
                raise RuntimeError(f"Không ghi được ảnh {target}")
            start = max(0, index / fps - 4)
            run_logged([
                "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
                "-i", str(preview), "-t", "8", "-an",
                "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264",
                "-crf", "25", "-preset", "fast", "-pix_fmt", "yuv420p",
                str(output / f"doan_{number}_tu_{start:.2f}s.mp4"),
            ], output / f"ffmpeg_{number}.log")
    finally:
        cap.release()


def write_report(output: Path, student: str, student_id: str, selections: dict,
                 records: list[dict]) -> None:
    """Tạo báo cáo có dữ kiện chạy thật và chừa phần quan sát cần xem video.

    Args:
        output: Thư mục gói kết quả.
        student: Họ tên sinh viên.
        student_id: Mã sinh viên.
        selections: Cấu hình được chọn hoặc tạm chọn theo video.
        records: Dữ kiện từng lần chạy và metric video_1.
    """
    lines = [
        "# Bản nháp báo cáo lab tracking", "",
        f"**Sinh viên:** {student} · **MSSV:** {student_id}", "",
        "Detector cố định: `yolo26n.pt`, ảnh 640 px, lớp người; "
        "Re-ID `osnet_x0_25_msmt17.pt`.", "",
        "Bản nháp tự điền cấu hình và số liệu thực tế. Cần xem minh chứng để hoàn thiện "
        "quan sát và xác nhận lựa chọn cho video_2–video_5 trước khi nộp.", "",
        "## 1. Cấu hình", "",
        "| Video | Cấu hình | Tracker | conf | iou | Cách chọn |",
        "|---|---|---|---|---|---|",
    ]
    for video in VIDEOS:
        chosen = selections[video]
        case = chosen["case"]
        lines.append(f"| {video} | {case['name']} | {case['tracker']} | {case['conf']} "
                     f"| {case['iou']} | {chosen['reason']} |")
    lines += ["", "## 2. Số liệu video_1", "",
              "Điểm theo thang 0–100 của TrackEval; MOTA có thể âm.", "",
              "| Cấu hình | HOTA | MOTA | IDF1 |", "|---|---|---|---|"]
    for record in records:
        if record["video"] == "video_1":
            metric = record["metrics"]
            lines.append(f"| {record['case']} | {metric['HOTA']:.3f} | "
                         f"{metric['MOTA']:.3f} | {metric['IDF1']:.3f} |")
    lines += ["", "video_2–video_5 không có nhãn; không tính metric cho các video này.",
              "", "## 3. Quan sát và cấu hình đã loại", ""]
    for video in VIDEOS:
        lines += [f"### {video}", "", "- Quan sát: [cần xem minh_chung và ghi frame/đoạn cụ thể].",
                  "- Cấu hình đã thử nhưng loại: [điền sau khi so sánh].", ""]
    lines += ["## 4. Phân tích", "",
              "[Viết 3–5 câu cho ít nhất hai video, liên hệ cảnh với lỗi tracking đã thấy.]",
              "", "## 5. Nếu có thêm thời gian", "", "[Điền sau khi xem kết quả.]", ""]
    (output / "BAO_CAO_nhap.md").write_text("\n".join(lines), encoding="utf-8")


def package_results(output: Path, student_id: str) -> Path:
    """Đóng gói kết quả, log và minh chứng; không đưa ảnh gốc hoặc trọng số vào.

    Args:
        output: Thư mục kết quả của luồng chạy.
        student_id: Mã sinh viên dùng cho tên ZIP.

    Returns:
        Đường dẫn ZIP để tải về và gửi lại.
    """
    target = output.parent / f"ket_qua_{student_id}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                relative = path.relative_to(output)
                if relative.parts[0] in {"cache", "thu_nghiem"}:
                    # Preview đầy đủ tải riêng; gói gửi lại dùng đoạn trích nhỏ hơn.
                    continue
                archive.write(path, relative.as_posix())
        for path in sorted((output / "thu_nghiem").rglob("*.txt")):
            archive.write(path, path.relative_to(output).as_posix())
        for path in sorted((output / "thu_nghiem").rglob("*_run.json")):
            archive.write(path, path.relative_to(output).as_posix())
    return target


def main() -> None:
    """Chạy ba cấu hình cho mỗi video và xuất gói tải về sau một lần chạy.

    Raises:
        RuntimeError: Khi dữ liệu, GPU hoặc đầu ra không đạt yêu cầu.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-data-root", type=Path, required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/kaggle"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-root", type=Path, help="Thư mục cache tạm; mặc định nằm trong --out")
    parser.add_argument("--student", default="Lưu Quang Khải")
    parser.add_argument("--student-id", default="2A202602599")
    args = parser.parse_args()
    args.lab_data_root = args.lab_data_root.resolve()
    args.trackeval_root = args.trackeval_root.resolve()
    args.out = args.out.resolve()
    if not args.student_id.isalnum():
        raise ValueError("Mã sinh viên chỉ dùng chữ và số.")
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("Thư mục kết quả đã có dữ liệu; chọn --out mới để tránh trộn lần chạy.")
    args.out.mkdir(parents=True, exist_ok=True)
    if not check(args.lab_data_root):
        raise RuntimeError("Gói dữ liệu chưa đủ.")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Chưa có GPU. Bật GPU trong Settings trước Save & Run All.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Thiếu ffmpeg để tạo minh chứng.")
    config = _load_eval_config(args.lab_data_root)
    scripts = Path(__file__).resolve().parent
    output = args.out
    environment = {
        "python": sys.version, "torch": torch.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device))
               if args.device.startswith("cuda") else "CPU",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(scripts.glob("*.py"))},
    }
    run_logged([sys.executable, "-m", "pip", "freeze"], output / "logs/packages.txt")
    environment["trackeval_commit"] = subprocess.check_output(
        ["git", "-C", str(args.trackeval_root), "rev-parse", "HEAD"], text=True
    ).strip()
    records = []
    selections = {}
    try:
        for video in VIDEOS:
            count, fps = sequence_info(args.lab_data_root, video)
            for case in DEFAULT_CASES:
                name = case["name"]
                directory = output / "thu_nghiem" / name
                cache_root = args.cache_root or output / "cache"
                cache = cache_root / f"{video}_c{case['conf']}_i{case['iou']}.jsonl"
                run_logged([
                    sys.executable, str(scripts / "run_tracking.py"),
                    "--source", str(args.lab_data_root / video / "img1"), "--seq-name", video,
                    "--tracker", case["tracker"], "--conf", str(case["conf"]),
                    "--iou", str(case["iou"]), "--device", args.device,
                    "--out", str(directory), "--save-video", "--fps", str(fps),
                    "--preview-width", "1280",
                    "--detection-cache", str(cache),
                ], output / "logs" / f"{video}_{name}.log")
                summary = json.loads((directory / f"{video}_run.json").read_text(encoding="utf-8"))
                if summary["frames_processed"] != count or summary["max_frames"] != 0:
                    raise RuntimeError(f"{video}/{name}: chưa chạy đủ frame.")
                record = {"video": video, "case": name, "images_expected": count, **summary}
                if video == "video_1":
                    run_name = f"{args.student_id}_{name}"
                    run_logged([
                        sys.executable, str(scripts / "evaluate_practice.py"),
                        "--trackeval-root", str(args.trackeval_root),
                        "--lab-data-root", str(args.lab_data_root),
                        "--submission", str(directory / "video_1.txt"), "--run-name", run_name,
                    ], output / "logs" / f"danh_gia_{name}.log")
                    evaluated = (args.trackeval_root / "data/trackers/mot_challenge"
                                 / f"{config['benchmark']}-{config.get('split', 'train')}" / run_name)
                    record["metrics"] = parse_summary(
                        (evaluated / "pedestrian_summary.txt").read_text(encoding="utf-8")
                    )
                    metric_dir = output / "metrics" / name
                    metric_dir.mkdir(parents=True, exist_ok=True)
                    for path in evaluated.glob("pedestrian_*.txt"):
                        shutil.copy2(path, metric_dir / path.name)
                    for path in evaluated.glob("pedestrian_*.csv"):
                        shutil.copy2(path, metric_dir / path.name)
                records.append(record)
                (output / "experiment_progress.json").write_text(
                    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                make_evidence(directory / f"{video}_preview.mp4",
                              output / "minh_chung" / video / name, count, fps)
            if video == "video_1":
                chosen = best_practice_case({r["case"]: r["metrics"] for r in records
                                             if r["video"] == video})
                reason = "HOTA cao nhất trong ba cấu hình; hòa điểm xét IDF1 rồi MOTA"
            else:
                chosen = "botsort_c030_i050"
                reason = "Tạm chọn; cần xem minh chứng để xác nhận hoặc đổi cấu hình"
            case = next(case for case in DEFAULT_CASES if case["name"] == chosen)
            selections[video] = {"case": case, "reason": reason,
                                 "requires_visual_review": video != "video_1"}
            submitted = output / "nop_bai"
            submitted.mkdir(exist_ok=True)
            for suffix in (".txt", "_run.json"):
                shutil.copy2(output / "thu_nghiem" / chosen / f"{video}{suffix}",
                             submitted / f"{video}{suffix}")
        manifest = {
            "status": "complete", "student": args.student, "student_id": args.student_id,
            "environment": environment, "cases": DEFAULT_CASES,
            "records": records, "selections": selections,
            "report_requires_visual_review": True,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output / "ket_qua_thu_nghiem.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["video", "case", "tracker", "conf", "iou", "frames_processed",
                      "elapsed_seconds", "fps", "detection_cache_hit", "HOTA", "MOTA", "IDF1"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow({**record, **record.get("metrics", {})})
        write_report(output, args.student, args.student_id, selections, records)
        (output / "DOC_TOI_TRUOC.md").write_text(
            "# Kết quả lab\n\n"
            "nop_bai/: 5 file đủ frame; video_2–video_5 đang tạm chọn BoT-SORT.\n\n"
            "thu_nghiem/: file TXT đủ frame của cả ba cấu hình, có thể đổi lựa chọn không chạy lại.\n\n"
            "minh_chung/: ba ảnh và ba đoạn 8 giây/cấu hình/video tại 10%, 50%, 90%. "
            "Đoạn cuối có thể ngắn hơn khi video ngắn. Đây chỉ là mẫu; xem preview đầy đủ "
            "trong output Kaggle nếu cần kiểm tra đoạn khác.\n\n"
            "metrics/: số liệu chính thức video_1. logs/ và manifest.json: dữ kiện tái kiểm tra.\n\n"
            "BAO_CAO_nhap.md cần hoàn thiện quan sát và phân tích trước khi nộp. "
            "Gửi ZIP kết quả lại để xem minh chứng, chọn cấu hình và viết báo cáo.\n",
            encoding="utf-8",
        )
        target = package_results(output, args.student_id)
        print(f"\nHoàn tất chạy và đóng gói: {target}", flush=True)
        print("Cần xem minh chứng để chốt bốn video không nhãn và hoàn thiện báo cáo.")
    except Exception as exc:
        (output / "manifest.json").write_text(
            json.dumps({"status": "failed", "error": str(exc), "records": records,
                        "selections": selections}, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (output / "LOI.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        package_results(output, args.student_id)
        raise


if __name__ == "__main__":
    main()
