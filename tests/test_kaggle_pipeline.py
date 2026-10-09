"""Kiểm tra luồng Kaggle bằng subprocess giả, không cần ảnh, GPU hay mạng."""

import json
import sys
import types
import subprocess
import zipfile
from pathlib import Path

import pytest

import run_kaggle_lab as pipeline
from evaluate_practice import stage, run_trackeval


def test_run_logged_persists_custom_environment_and_failure(tmp_path: Path) -> None:
    log = tmp_path / "working/logs_cai_dat/kiem_tra.log"
    pipeline.run_logged([sys.executable, "-c", "import os; print(os.environ['LAB_TEST_MESSAGE'])"],
                        log, {"LAB_TEST_MESSAGE": "da luu log"})
    assert "da luu log" in log.read_text(encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_logged([sys.executable, "-c", "print('loi mau'); raise SystemExit(2)"], log)
    assert "loi mau" in log.read_text(encoding="utf-8")


def make_lab(root: Path) -> None:
    for video in pipeline.VIDEOS:
        images = root / video / "img1"
        images.mkdir(parents=True)
        for index in (1, 2):
            (images / f"{index:06}.jpg").write_bytes(b"anh gia")
    practice = root / "video_1"
    (practice / "gt").mkdir()
    (practice / "gt/gt.txt").write_text("1,1,0,0,10,10,1,1,1\n")
    (practice / "eval_config.json").write_text('{"benchmark": "LAB", "split": "train"}')
    (practice / "seqinfo.ini").write_text("[Sequence]\nseqLength=2\nframeRate=7.5\n")


def test_sequence_info_checks_count_and_preserves_fps(tmp_path: Path) -> None:
    make_lab(tmp_path)
    assert pipeline.sequence_info(tmp_path, "video_1") == (2, 7.5)
    assert pipeline.sequence_info(tmp_path, "video_2") == (2, 20.0)
    (tmp_path / "video_1/img1/000002.jpg").unlink()
    with pytest.raises(ValueError, match="seqLength"):
        pipeline.sequence_info(tmp_path, "video_1")


def test_stage_uses_configured_split(tmp_path: Path) -> None:
    lab = tmp_path / "lab"
    make_lab(lab)
    submission = tmp_path / "video_1.txt"
    submission.write_text("1,1,0,0,10,10,0.9,-1,-1,-1\n")
    evaluator = tmp_path / "TrackEval"
    stage(evaluator, lab, submission, "thu", "LAB", "test")
    assert (evaluator / "data/gt/mot_challenge/LAB-test/video_1/gt/gt.txt").exists()
    assert (evaluator / "data/trackers/mot_challenge/LAB-test/thu/data/video_1.txt").exists()
    assert not (evaluator / "data/gt/mot_challenge/LAB-train").exists()


def test_trackeval_patches_aliases_in_child_and_rejects_missing_summary(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("evaluate_practice.subprocess.run", lambda command, check: calls.append(command))
    with pytest.raises(RuntimeError, match="summary"):
        run_trackeval(tmp_path, "thu", "LAB", "train")
    assert calls[0][1] == "-c"
    assert "np.float = float" in calls[0][2]
    assert "np.int = int" in calls[0][2]
    assert calls[0][-2:] == ["--PLOT_CURVES", "False"]


def test_main_runs_all_cases_and_packages_reviewable_results(tmp_path, monkeypatch) -> None:
    lab = tmp_path / "lab"
    make_lab(lab)
    output = tmp_path / "results"
    evaluator = tmp_path / "TrackEval"
    commands = []

    def fake_logged(command, log_path):
        commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log gia\n", encoding="utf-8")
        if any(arg.endswith("run_tracking.py") for arg in command):
            option = lambda name: command[command.index(name) + 1]
            directory = Path(option("--out"))
            directory.mkdir(parents=True, exist_ok=True)
            video = option("--seq-name")
            (directory / f"{video}.txt").write_text("1,1,0,0,10,10,0.9,-1,-1,-1\n")
            (directory / f"{video}_preview.mp4").write_bytes(b"video gia")
            (directory / f"{video}_run.json").write_text(json.dumps({
                "frames_processed": 2, "max_frames": 0, "tracker": option("--tracker"),
                "conf": float(option("--conf")), "iou": float(option("--iou")),
                "fps": 10, "elapsed_seconds": 0.2,
            }))
        elif any(arg.endswith("evaluate_practice.py") for arg in command):
            assert Path(command[command.index("--submission") + 1]).name == "video_1.txt"
            run = command[command.index("--run-name") + 1]
            directory = evaluator / "data/trackers/mot_challenge/LAB-train" / run
            directory.mkdir(parents=True, exist_ok=True)
            score = 65 if "c015" in run else 60
            (directory / "pedestrian_summary.txt").write_text(f"HOTA MOTA IDF1\n{score} 70 80\n")

    def fake_evidence(preview, output, count, fps):
        output.mkdir(parents=True)
        (output / "mau.jpg").write_bytes(b"anh minh chung")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="gia"))
    monkeypatch.setattr(pipeline, "run_logged", fake_logged)
    monkeypatch.setattr(pipeline, "make_evidence", fake_evidence)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda *args, **kwargs: "commit-gia\n")
    monkeypatch.setattr(sys, "argv", ["run_kaggle_lab.py", "--lab-data-root", str(lab),
                                     "--trackeval-root", str(evaluator), "--device", "cpu", "--out", str(output)])
    pipeline.main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["records"]) == 15
    assert manifest["selections"]["video_1"]["case"]["name"] == "botsort_c015_i050"
    assert all(manifest["selections"][video]["requires_visual_review"] for video in pipeline.VIDEOS[1:])
    tracking = [c for c in commands if any(arg.endswith("run_tracking.py") for arg in c)]
    evaluations = [c for c in commands if any(arg.endswith("evaluate_practice.py") for arg in c)]
    assert len(tracking) == 15 and len(evaluations) == 3
    assert all("--max-frames" not in c for c in tracking)
    assert tracking[0][tracking[0].index("--detection-cache") + 1] == tracking[1][tracking[1].index("--detection-cache") + 1]
    assert tracking[1][tracking[1].index("--detection-cache") + 1] != tracking[2][tracking[2].index("--detection-cache") + 1]
    with zipfile.ZipFile(output.parent / "ket_qua_2A202602599.zip") as archive:
        names = archive.namelist()
        assert sum(name.startswith("nop_bai/") and name.endswith(".txt") for name in names) == 5
        assert sum(name.startswith("thu_nghiem/") and name.endswith(".txt") for name in names) == 15
        assert not any(name.endswith("_preview.mp4") for name in names)
        assert "BAO_CAO_nhap.md" in names
        assert "minh_chung/video_5/botsort_c015_i050/mau.jpg" in names
    report = (output / "BAO_CAO_nhap.md").read_text(encoding="utf-8")
    assert "Lưu Quang Khải" in report and "2A202602599" in report
    assert "cần xem minh_chung" in report


def test_package_excludes_detection_cache_and_full_previews(tmp_path) -> None:
    output = tmp_path / "results"
    for name in ("cache/video_1.jsonl", "thu_nghiem/thu/video_1_preview.mp4",
                 "thu_nghiem/thu/video_1.txt", "manifest.json"):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("gia")
    with zipfile.ZipFile(pipeline.package_results(output, "2A202602599")) as archive:
        assert set(archive.namelist()) == {"manifest.json", "thu_nghiem/thu/video_1.txt"}


def test_notebook_is_clean_and_code_cells_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "kaggle_run_all.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), "kaggle_run_all.ipynb", "exec")
