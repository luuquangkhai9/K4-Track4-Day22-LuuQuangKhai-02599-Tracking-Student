"""Quản lý cache detection và đọc số liệu thực nghiệm không cần GPU."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def detection_metadata(source: Path, conf: float, iou: float, max_frames: int = 0) -> dict:
    """Tạo dấu nhận diện dữ liệu và cấu hình để tránh dùng nhầm cache.

    Args:
        source: Thư mục ảnh đầu vào.
        conf: Ngưỡng confidence của detector.
        iou: Ngưỡng IoU của detector.
        max_frames: Giới hạn frame; 0 nghĩa là toàn bộ.

    Returns:
        Metadata gồm cấu hình cố định và dấu nhận diện danh sách ảnh.

    Raises:
        ValueError: Khi đầu vào không phải thư mục ảnh hoặc không có ảnh.
    """
    if not source.is_dir():
        raise ValueError("Cache detection chỉ hỗ trợ thư mục img1.")
    paths = sorted(source.glob("*.jpg"))
    if max_frames:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"Không có ảnh trong {source}")
    signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    return {
        "version": 1, "source": str(source.resolve()), "frames": len(paths),
        "images_signature": digest, "detector": "yolo26n.pt", "imgsz": 640,
        "class_id": 0, "conf": conf, "iou": iou,
    }


class DetectionCache:
    """Đọc/ghi cache theo frame; chỉ công nhận cache đã ghi đủ và thành công."""

    def __init__(self, path: Path, metadata: dict):
        """Mở cache đã có hoặc file tạm cho lần chạy đầu.

        Args:
            path: Đường dẫn file JSONL cache.
            metadata: Dấu nhận diện dữ liệu và cấu hình detector.

        Raises:
            ValueError: Khi metadata cache không khớp lần chạy hiện tại.
        """
        self.path = path
        self.metadata = metadata
        self.hit = path.exists()
        self.count = 0
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = (path if self.hit else self.temporary).open(
            "r" if self.hit else "w", encoding="utf-8"
        )
        if self.hit:
            try:
                if json.loads(self.stream.readline()) != metadata:
                    raise ValueError("Cache không khớp ảnh hoặc cấu hình; dùng đường dẫn cache mới.")
            except Exception:
                self.stream.close()
                raise
        else:
            self.stream.write(json.dumps(metadata) + "\n")

    def frame(self, index: int, detections: list | None = None) -> list:
        """Đọc hoặc ghi các hộp detection của một frame theo đúng thứ tự.

        Args:
            index: Số thứ tự frame, bắt đầu từ 0.
            detections: Danh sách hộp khi ghi cache mới.

        Returns:
            Danh sách hộp detection của frame.

        Raises:
            ValueError: Khi frame sai thứ tự, thiếu hộp để ghi hoặc cache hỏng.
        """
        if index != self.count:
            raise ValueError("Frame cache không liên tiếp.")
        if self.hit:
            record = json.loads(self.stream.readline())
            if record["frame"] != index:
                raise ValueError("Sai thứ tự frame trong cache.")
            detections = record["detections"]
        elif detections is None:
            raise ValueError("Thiếu detection để ghi cache.")
        else:
            self.stream.write(json.dumps({"frame": index, "detections": detections}) + "\n")
        self.count += 1
        return detections

    def close(self, success: bool) -> None:
        """Đóng file và chỉ công nhận cache mới nếu đã xử lý đủ frame.

        Args:
            success: Lần chạy tracking có thành công hay không.

        Raises:
            ValueError: Khi số frame cache thiếu hoặc thừa.
        """
        valid = self.count == self.metadata["frames"]
        if self.hit and success and valid:
            valid = not self.stream.readline()
        self.stream.close()
        if not self.hit:
            if success and valid:
                self.temporary.replace(self.path)
            else:
                self.temporary.unlink(missing_ok=True)
        if success and not valid:
            raise ValueError("Cache chưa đủ frame hoặc có dữ liệu thừa.")


def parse_summary(text: str) -> dict[str, float]:
    """Đọc HOTA, MOTA và IDF1 từ file summary chính thức của TrackEval.

    Args:
        text: Nội dung pedestrian_summary.txt, gồm dòng tiêu đề và dòng số.

    Returns:
        Ba metric theo thang điểm do TrackEval xuất ra.

    Raises:
        ValueError: Khi thiếu metric, sai định dạng hoặc có số không hữu hạn.
    """
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if len(lines) != 2 or len(lines[0]) != len(lines[1]):
        raise ValueError("Summary TrackEval không đúng định dạng hai dòng.")
    values = dict(zip(lines[0], lines[1]))
    try:
        result = {key: float(values[key]) for key in ("HOTA", "MOTA", "IDF1")}
    except (KeyError, ValueError) as exc:
        raise ValueError("Summary thiếu HOTA/MOTA/IDF1 hoặc giá trị không hợp lệ.") from exc
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Metric không hữu hạn.")
    return result


def best_practice_case(metrics: dict[str, dict[str, float]]) -> str:
    """Chọn cấu hình video_1 theo HOTA, rồi IDF1, MOTA và tên khi hòa điểm.

    Args:
        metrics: Metric theo tên cấu hình đã chạy đủ frame trên video_1.

    Returns:
        Tên cấu hình có điểm ưu tiên cao nhất.

    Raises:
        ValueError: Khi không có kết quả đánh giá.
    """
    if not metrics:
        raise ValueError("Không có metric để chọn cấu hình video_1.")
    return max(sorted(metrics), key=lambda name: (
        metrics[name]["HOTA"], metrics[name]["IDF1"], metrics[name]["MOTA"]
    ))
