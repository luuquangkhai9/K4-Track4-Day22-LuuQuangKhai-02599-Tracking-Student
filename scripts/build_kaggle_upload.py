"""Tạo gói mã nguồn nhỏ để gắn vào Dataset Kaggle, không chứa dữ liệu lab."""

from pathlib import Path
import zipfile


def main() -> None:
    """Đóng gói mã nguồn, notebook và hướng dẫn vào runs/kaggle_upload.zip."""
    root = Path(__file__).resolve().parents[1]
    target = root / "runs/kaggle_upload.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = [root / name for name in (
        "README.md", "HUONG_DAN.md", "HUONG_DAN_KAGGLE.md", "AGENTS.md",
        "environment.yml", "requirements.txt", "requirements-kaggle.txt",
        "pytest.ini", "kaggle_run_all.ipynb", "on_tap_metrics.ipynb",
    )]
    for directory, suffix in (("scripts", ".py"), ("tests", ".py"), ("submission_template", ".md")):
        paths.extend(sorted((root / directory).glob(f"*{suffix}")))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    print(f"Gói mã nguồn để tải lên Kaggle: {target}")
    print("Tải gói này cùng dữ liệu lab lên Dataset riêng tư và import kaggle_run_all.ipynb.")


if __name__ == "__main__":
    main()
