"""Tạo notebook Kaggle tự cài đặt và chạy lab trong một phiên Save & Run All."""

import json
from pathlib import Path


def main() -> None:
    """Ghi notebook UTF-8 với cấu hình sinh viên và luồng chạy tuần tự."""
    cells = []
    blocks = [
        ("markdown", '''# Lab tracking — một lần Save & Run All

**Lưu Quang Khải · MSSV 2A202602599**

Import notebook này, gắn Input chứa `kaggle_upload.zip` và dữ liệu lab, bật **GPU + Internet**,
rồi **Save Version → Save & Run All**. Xem `HUONG_DAN_KAGGLE.md` trước khi chạy.
Notebook tự phát hiện đường dẫn khi Input có đúng một bản repo và một bộ lab.

Luồng chạy đủ frame cho ba cấu hình/video, chấm **chỉ video_1**, xuất TXT của mọi
cấu hình và minh chứng. Bốn video không nhãn cần xem để chốt lựa chọn; báo cáo nháp
chưa tự tạo quan sát. Tải `ket_qua_2A202602599.zip` ở Output và gửi lại để hoàn thiện báo cáo.
'''),
        ("code", '''from pathlib import Path
import os
import sys
import subprocess
import shutil
import zipfile

STUDENT = "Lưu Quang Khải"
STUDENT_ID = "2A202602599"
REPO_HINT = ""  # Đường dẫn repo đã giải nén; để trống để tự tìm.
LAB_DATA_HINT = ""  # Thư mục chứa video_1…video_5; để trống để tự tìm.
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/tracking_lab")
TEMP.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
if not INPUT.exists():
    raise RuntimeError("Notebook này cần chạy trên Kaggle và có Dataset trong Input.")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
# BoxMOT cũ cần cách nạp checkpoint Re-ID cố định theo hành vi torch.load trước đây.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
'''),
        ("markdown", "## Tìm mã nguồn và dữ liệu\n\nChỉ giải nén ZIP có cấu trúc repo hoặc lab; không tải ảnh lên nguồn công khai."),
        ("code", '''extracted = TEMP / "inputs"
extracted.mkdir(exist_ok=True)
direct_repos = [path for path in INPUT.rglob("run_kaggle_lab.py") if path.parent.name == "scripts"]
direct_labs = [path for path in INPUT.rglob("video_1")
               if path.is_dir() and (path / "gt/gt.txt").is_file()
               and all(any(image.is_file() for image in
                           (path.parent / f"video_{i}" / "img1").glob("*.jpg"))
                       for i in range(1, 6))]
for number, archive in enumerate(sorted(INPUT.rglob("*.zip"))):
    with zipfile.ZipFile(archive) as source:
        names = [item.filename.replace("\\\\", "/") for item in source.infolist()]
        is_repo = not direct_repos and any(name.endswith("scripts/run_kaggle_lab.py") for name in names)
        is_lab = not direct_labs and any("video_1/img1/" in name for name in names)
        if not (is_repo or is_lab):
            continue
        destination = extracted / f"goi_{number}"
        destination.mkdir(exist_ok=True)
        for name in names:
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError(f"ZIP có đường dẫn không hợp lệ: {archive}")
        source.extractall(destination)

roots = [INPUT, extracted]
repos = sorted({path.parents[1] for root in roots for path in root.rglob("run_kaggle_lab.py")
                if path.parent.name == "scripts"})
if REPO_HINT:
    repos = [Path(REPO_HINT)]
if len(repos) != 1:
    raise RuntimeError(f"Cần đúng một repo mã nguồn. Repo: {repos}. Kiểm tra Add Input hoặc REPO_HINT.")
REPO = TEMP / "repo"
shutil.copytree(repos[0], REPO, dirs_exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from check_data import discover_data_roots, describe_data_root, usable_data_root
import json
labs = discover_data_roots(roots)
if LAB_DATA_HINT:
    if not usable_data_root(Path(LAB_DATA_HINT)):
        print(json.dumps(describe_data_root(Path(LAB_DATA_HINT)), ensure_ascii=False, indent=2))
        print("Các thư mục hợp lệ tìm được:", labs)
        raise RuntimeError("LAB_DATA_HINT không có đủ ảnh/nhãn. Chọn đường dẫn thật có file dữ liệu.")
    labs = [Path(LAB_DATA_HINT)]
if len(labs) != 1:
    print("Các ZIP trong Input:", [str(path) for path in INPUT.rglob("*.zip")])
    for root in roots:
        for video in root.rglob("video_1"):
            if video.is_dir():
                print(json.dumps(describe_data_root(video.parent), ensure_ascii=False, indent=2))
    raise RuntimeError(f"Cần đúng một bộ lab có file ảnh và nhãn. Tìm được: {labs}. "
                       "Kiểm tra Dataset/phiên bản Input, giải nén ZIP hoặc điền LAB_DATA_HINT.")
LAB_DATA = labs[0]
os.environ["LAB_DATA"] = str(LAB_DATA)
os.chdir(REPO)
print("Repo:", REPO, "\\nDữ liệu:", LAB_DATA)
'''),
        ("markdown", "## Cài Python 3.11 riêng, PyTorch GPU và TrackEval\n\nKernel Kaggle có thể là Python 3.13. Luồng lab dùng Python 3.11 riêng do uv tải về, không dùng thư viện Python 3.13 của kernel. Cần Internet để tải Python và PyTorch CUDA."),
        ("code", '''print("Python của kernel Kaggle:", sys.version)
from run_kaggle_lab import run_logged
SETUP_LOGS = WORK / "logs_cai_dat"
run_logged(["nvidia-smi", "-L"], SETUP_LOGS / "00_gpu.log")
UV_TOOLS = TEMP / "uv_tools"
run_logged([sys.executable, "-m", "pip", "install", "--upgrade",
            "--target", str(UV_TOOLS), "uv<1"], SETUP_LOGS / "01_uv.log")
uv_env = dict(os.environ)
uv_env["PYTHONPATH"] = str(UV_TOOLS)
uv_env["UV_PYTHON_INSTALL_DIR"] = str(TEMP / "pythons")
uv_env["UV_CACHE_DIR"] = str(TEMP / "uv_cache")
uv_command = [sys.executable, "-m", "uv"]
run_logged(uv_command + ["python", "install", "3.11"], SETUP_LOGS / "02_python.log", uv_env)
VENV = TEMP / "venv_py311"
run_logged(uv_command + ["venv", "--python", "3.11", "--seed", "--allow-existing",
                         str(VENV)], SETUP_LOGS / "03_venv.log", uv_env)
PYTHON = VENV / "bin/python"
run_logged([str(PYTHON), "--version"], SETUP_LOGS / "04_phien_ban_python.log")
run_logged([str(PYTHON), "-m", "pip", "install", "--upgrade", "pip"], SETUP_LOGS / "05_pip.log")
run_logged([str(PYTHON), "-m", "pip", "install", "torch==2.6.0", "torchvision==0.21.0",
            "--index-url", "https://download.pytorch.org/whl/cu124"], SETUP_LOGS / "06_pytorch.log")
run_logged([str(PYTHON), "-m", "pip", "install", "-r", str(REPO / "requirements-kaggle.txt")],
           SETUP_LOGS / "07_thu_vien.log")
run_logged([str(PYTHON), "-m", "pip", "install", "--no-deps", "boxmot==10.0.42"],
           SETUP_LOGS / "08_boxmot.log")
TRACKEVAL = TEMP / "TrackEval"
if not TRACKEVAL.exists():
    run_logged(["git", "clone", "--depth", "1", "https://github.com/JonathonLuiten/TrackEval.git",
                str(TRACKEVAL)], SETUP_LOGS / "09_clone_trackeval.log")
run_logged(["git", "-C", str(TRACKEVAL), "fetch", "--depth", "1", "origin", TRACKEVAL_COMMIT],
           SETUP_LOGS / "10_fetch_trackeval.log")
run_logged(["git", "-C", str(TRACKEVAL), "checkout", TRACKEVAL_COMMIT],
           SETUP_LOGS / "11_checkout_trackeval.log")
run_logged([str(PYTHON), "-m", "pip", "install", "--no-deps", "-e", str(TRACKEVAL)],
           SETUP_LOGS / "12_cai_trackeval.log")
if not shutil.which("ffmpeg"):
    raise RuntimeError("Image Kaggle thiếu ffmpeg. Chọn image có ffmpeg trước khi chạy luồng này.")
'''),
        ("markdown", "## Kiểm tra toàn bộ luồng trên mẫu nhỏ — 20 frame/video\n\nChạy cả 15 lượt trên bản mẫu riêng, kiểm tra cache, preview/ffmpeg, đánh giá video_1 và đóng ZIP. Chỉ chạy toàn bộ khi bước này đạt. Số liệu mẫu không dùng cho báo cáo chính."),
        ("code", '''subprocess.run([str(PYTHON), "-c",
    "import torch, numpy, cv2, ultralytics, boxmot; "
    "assert torch.cuda.is_available(), 'Chưa bật GPU trong Settings'; "
    "print('Thử tính toán CUDA:', (torch.ones(2, device='cuda') * 2).sum().item()); "
    "print('GPU:', torch.cuda.get_device_name(0)); "
    "print('NumPy:', numpy.__version__, 'YOLO:', ultralytics.__version__, 'BoxMOT:', boxmot.__version__)"], check=True)
subprocess.run([str(PYTHON), "-m", "pytest", "-q"], check=True)
subprocess.run([str(PYTHON), "scripts/check_data.py", "--lab-data-root", str(LAB_DATA)], check=True)
subprocess.run([str(PYTHON), "-c",
    "from pathlib import Path; import gdown; "
    "from boxmot.appearance.reid_model_factory import get_model_url; "
    "weights = Path('osnet_x0_25_msmt17.pt'); "
    "weights.is_file() or gdown.download(get_model_url(weights), str(weights), quiet=False); "
    "assert weights.is_file() and weights.stat().st_size > 1024, 'Không tải được trọng số Re-ID cố định'"], check=True)
SMOKE_ROOT = WORK / "kiem_tra_mau"
run_logged([str(PYTHON), "scripts/smoke_test_lab.py",
    "--lab-data-root", str(LAB_DATA), "--trackeval-root", str(TRACKEVAL),
    "--work-root", str(SMOKE_ROOT), "--sample-root", str(TEMP / "data_mau"),
    "--cache-root", str(TEMP / "cache_mau"), "--frames", "20", "--device", "cuda:0"],
    WORK / "logs_chay/kiem_tra_mau.log")
assert (SMOKE_ROOT / "DAT_KIEM_TRA.json").is_file(), "Mẫu chưa đạt; không chạy dữ liệu đầy đủ"
'''),
        ("markdown", "## Chạy đầy đủ và đóng gói\n\nKhông giới hạn frame trong bước này. Luồng sẽ dừng nếu bất kỳ lượt chạy hoặc đánh giá nào lỗi."),
        ("code", '''assert (SMOKE_ROOT / "DAT_KIEM_TRA.json").is_file(), "Cần chạy cell kiểm tra mẫu trước"
OUTPUT = WORK / "ket_qua_lab"
run_logged([str(PYTHON), "scripts/run_kaggle_lab.py",
    "--lab-data-root", str(LAB_DATA), "--trackeval-root", str(TRACKEVAL),
    "--out", str(OUTPUT), "--cache-root", str(TEMP / "cache_day_du"),
    "--device", "cuda:0", "--student", STUDENT, "--student-id", STUDENT_ID],
    WORK / "logs_chay/toan_bo.log")
'''),
        ("markdown", "## Tải kết quả\n\nDùng Output của phiên Save & Run All đã lưu. Gửi ZIP để xem minh chứng và hoàn thiện báo cáo; kiểm tra lựa chọn tạm thời trước khi nộp."),
        ("code", '''import json
from IPython.display import FileLink, display

manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete"
archive = WORK / f"ket_qua_{STUDENT_ID}.zip"
assert archive.exists()
print(f"Đã chạy đủ {len(manifest['records'])} lượt. Gói tải về: {archive}")
print(f"Dung lượng ZIP: {archive.stat().st_size / 1024**2:.1f} MB")
print("Bốn video không nhãn đang tạm chọn BoT-SORT; cần xem minh chứng để chốt.")
os.chdir(WORK)
display(FileLink(archive.name))
print((OUTPUT / "BAO_CAO_nhap.md").read_text(encoding="utf-8"))
'''),
    ]
    for kind, text in blocks:
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "cells": cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }, "nbformat": 4, "nbformat_minor": 5,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"tracking-{index:02d}"
    target = Path(__file__).resolve().parents[1] / "kaggle_run_all.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Đã tạo notebook: {target}")


if __name__ == "__main__":
    main()
