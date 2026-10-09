#!/usr/bin/env python
"""Kiểm tra gói lab_data giảng viên phát đã đủ ảnh (và nhãn video luyện) chưa.

Ví dụ:
    python scripts/check_data.py --lab-data-root "$LAB_DATA"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIDEOS = ["video_1", "video_2", "video_3", "video_4", "video_5"]
PRACTICE_VIDEO = "video_1"

CONTEXT = {
    "video_1": "Quảng trường ngoài trời, camera TĨNH, ban ngày, mật độ trung bình — video luyện, có nhãn",
    "video_2": "Phố, camera TĨNH trên cao, BAN ĐÊM, mật độ RẤT đông",
    "video_3": "Camera DI CHUYỂN, độ phân giải thấp, khung hình chậm",
    "video_4": "TRONG NHÀ, camera di chuyển tiến về phía trước, phản chiếu kính",
    "video_5": "Quay từ XE BUS tại giao lộ đông, rung lắc mạnh",
}


def usable_data_root(root: Path) -> bool:
    """Kiểm tra thư mục có ảnh cho đủ năm video và nhãn video_1 hay không.

    Args:
        root: Thư mục ứng viên chứa video_1 đến video_5.

    Returns:
        True nếu có ít nhất một ảnh JPG mỗi video và file nhãn video_1.
    """
    return all(any(path.is_file() for path in (root / name / "img1").glob("*.jpg"))
               for name in VIDEOS) and (root / "video_1/gt/gt.txt").is_file()


def discover_data_roots(search_roots: list[Path]) -> list[Path]:
    """Tìm các thư mục dữ liệu thực sự có ảnh và nhãn, bỏ thư mục rỗng.

    Args:
        search_roots: Các vị trí chứa dữ liệu gắn vào notebook hoặc đã giải nén.

    Returns:
        Danh sách thư mục hợp lệ, không trùng nhau và sắp xếp theo đường dẫn.
    """
    candidates = {path.parent.resolve() for root in search_roots
                  for path in root.rglob("video_1") if path.is_dir()}
    return sorted(root for root in candidates if usable_data_root(root))


def describe_data_root(root: Path) -> dict:
    """Ghi thông tin cấu trúc thật để chẩn đoán sai đường dẫn hoặc thiếu file.

    Args:
        root: Thư mục dữ liệu đang được chọn.

    Returns:
        Đường dẫn, trạng thái tồn tại, số ảnh và vài tên file trong mỗi img1.
    """
    videos = {}
    for video in VIDEOS:
        images = root / video / "img1"
        children = sorted(images.iterdir()) if images.is_dir() else []
        videos[video] = {
            "img1": str(images), "img1_exists": images.is_dir(),
            "jpg_files": sum(path.is_file() and path.suffix == ".jpg" for path in children),
            "sample_entries": [path.name + ("/" if path.is_dir() else "") for path in children[:8]],
            "gt_exists": (root / video / "gt/gt.txt").is_file(),
        }
    return {"root": str(root), "root_exists": root.is_dir(), "videos": videos}


def check(lab_data_root: Path) -> bool:
    """In trạng thái từng video và báo gói lab có dùng được không.

    Args:
        lab_data_root: Thư mục lab_data giảng viên phát.

    Returns:
        True nếu cả năm video có ảnh và video_1 có file nhãn.
    """
    all_ok = True
    for name in VIDEOS:
        img_dir = lab_data_root / name / "img1"
        n_imgs = sum(path.is_file() for path in img_dir.glob("*.jpg")) if img_dir.exists() else 0
        gt_file = lab_data_root / name / "gt" / "gt.txt"
        has_gt = gt_file.exists()
        if name == PRACTICE_VIDEO:
            ok = n_imgs > 0 and has_gt
            gt_note = "có nhãn" if has_gt else "THIẾU nhãn video luyện"
        else:
            ok = n_imgs > 0
            gt_note = "không có nhãn (đúng)" if not has_gt else "có nhãn — không dùng để tự chấm"
        all_ok &= ok
        status = "OK   " if ok else "THIẾU"
        print(f"[{status}] {name:8s}  {n_imgs:4d} ảnh  {gt_note}  — {CONTEXT[name]}")
    preview_dir = lab_data_root / "preview"
    n_preview = len(list(preview_dir.glob("*.mp4"))) if preview_dir.exists() else 0
    print(f"\nPreview .mp4: {n_preview} file trong {preview_dir}")
    if all_ok:
        print("Gói dữ liệu đủ để chạy scripts/run_tracking.py.")
    else:
        print("Không thấy đủ ảnh/nhãn tại đường dẫn này. Kiểm tra LAB_DATA, Input và ZIP đã giải nén.")
        print(json.dumps(describe_data_root(lab_data_root), ensure_ascii=False, indent=2))
    return all_ok


def main() -> None:
    """Đọc ``--lab-data-root`` và in kết quả kiểm tra gói dữ liệu."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-data-root", required=True, type=Path, help="Thư mục lab_data giảng viên phát")
    args = parser.parse_args()
    if not check(args.lab_data_root):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
