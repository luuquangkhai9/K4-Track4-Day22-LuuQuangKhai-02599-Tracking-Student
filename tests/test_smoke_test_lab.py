"""Kiểm tra tạo mẫu và luồng chạy thử mà không cần GPU, ảnh thật hoặc mạng."""

import configparser
import json
import sys
from pathlib import Path

import pytest

import smoke_test_lab as smoke
from evaluate_practice import _load_eval_config
from test_kaggle_pipeline import make_lab


def test_filter_ground_truth_crops_only_requested_frames() -> None:
    text = "1,2,0,0,10,20,1,1,1\n2,2,0,0,10,20,1,1,1\n3,2,0,0,10,20,1,1,1\n"
    assert smoke.filter_ground_truth(text, 2) == "\n".join(text.splitlines()[:2]) + "\n"
    assert smoke.filter_ground_truth("3,2,0,0,10,20,1,1,1\n", 2) == ""


@pytest.mark.parametrize("text,frames", [("1,2,3", 2), ("1.5,2,0,0,10,20,1,1,1", 2),
                                        ("a,2,0,0,10,20,1,1,1", 2), ("", 0)])
def test_filter_ground_truth_rejects_invalid_input(text, frames) -> None:
    with pytest.raises(ValueError):
        smoke.filter_ground_truth(text, frames)


def test_config_optional_for_original_folder_layout(tmp_path: Path) -> None:
    root = tmp_path / "data_lab21"
    make_lab(root)
    (root / "video_1/eval_config.json").unlink()
    (root / "video_1/det").mkdir()
    (root / "video_1/det/det.txt").write_text("khong dung detection co san")
    assert _load_eval_config(root) == {"benchmark": "LAB", "split": "train"}


@pytest.mark.parametrize("config", ['[]', '{"benchmark": "../ngoai"}',
                                    '{"benchmark": "LAB", "split": ""}', '{}'])
def test_eval_config_rejects_invalid_values(tmp_path, config) -> None:
    (tmp_path / "video_1").mkdir()
    (tmp_path / "video_1/eval_config.json").write_text(config)
    with pytest.raises(ValueError):
        _load_eval_config(tmp_path)


def test_sample_has_cropped_labels_and_does_not_change_original(tmp_path: Path) -> None:
    source = tmp_path / "data_lab21"
    make_lab(source)
    for video in smoke.VIDEOS:
        (source / video / "img1/000003.jpg").write_bytes(b"anh gia thu ba")
    info = source / "video_1/seqinfo.ini"
    info.write_text("[Sequence]\nseqLength=3\nframeRate=7.5\n")
    labels = source / "video_1/gt/gt.txt"
    labels.write_text("1,1,0,0,10,10,1,1,1\n3,1,0,0,10,10,1,1,1\n")
    original_info = info.read_bytes()
    original_labels = labels.read_bytes()
    (source / "video_1/eval_config.json").unlink()
    target = tmp_path / "sample"
    counts = smoke.create_sample(source, target, frames=2)
    assert counts == {video: 2 for video in smoke.VIDEOS}
    assert info.read_bytes() == original_info and labels.read_bytes() == original_labels
    assert (target / "video_1/gt/gt.txt").read_text() == "1,1,0,0,10,10,1,1,1\n"
    parser = configparser.ConfigParser()
    parser.read(target / "video_1/seqinfo.ini")
    assert parser["Sequence"].getint("seqLength") == 2
    assert parser["Sequence"].getfloat("frameRate") == 7.5
    assert not any((target / video / "gt").exists() for video in smoke.VIDEOS[1:])
    assert not (target / "video_1/det").exists()
    assert _load_eval_config(target) == {"benchmark": "LAB", "split": "train"}
    with pytest.raises(FileExistsError):
        smoke.create_sample(source, target)


def test_sample_rejects_non_contiguous_names(tmp_path: Path) -> None:
    source = tmp_path / "lab"
    make_lab(source)
    (source / "video_2/img1/000001.jpg").rename(source / "video_2/img1/000004.jpg")
    with pytest.raises(ValueError, match="liên tiếp"):
        smoke.create_sample(source, tmp_path / "sample")


@pytest.mark.parametrize("separate_sample", [False, True])
def test_smoke_main_only_marks_passed_after_entire_pipeline(tmp_path, monkeypatch, separate_sample) -> None:
    source = tmp_path / "data_lab21"
    make_lab(source)
    work = tmp_path / "work"
    commands = []

    def fake_logged(command, log_path):
        commands.append(command)
        output = Path(command[command.index("--out") + 1])
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps({"status": "complete", "records": [{}] * 15}))
        (output.parent / "ket_qua_SMOKE.zip").write_bytes(b"zip gia")

    monkeypatch.setattr(smoke, "run_logged", fake_logged)
    command = ["smoke_test_lab.py", "--lab-data-root", str(source),
               "--trackeval-root", str(tmp_path / "TrackEval"),
               "--work-root", str(work), "--frames", "2", "--device", "cpu"]
    sample = tmp_path / "temp/data_mau" if separate_sample else work / "data_mau"
    if separate_sample:
        command.extend(["--sample-root", str(sample), "--cache-root", str(tmp_path / "temp/cache")])
    monkeypatch.setattr(sys, "argv", command)
    smoke.main()
    assert len(commands) == 1
    assert commands[0][commands[0].index("--lab-data-root") + 1] == str(sample)
    if separate_sample:
        assert not (work / "data_mau").exists()
        assert commands[0][commands[0].index("--cache-root") + 1] == str(tmp_path / "temp/cache")
    assert commands[0][commands[0].index("--student-id") + 1] == "SMOKE"
    assert json.loads((work / "DAT_KIEM_TRA.json").read_text(encoding="utf-8"))["status"] == "passed"


def test_notebook_blocks_full_run_until_sample_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "kaggle_run_all.ipynb").read_text(encoding="utf-8"))
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    sample = next(index for index, code in enumerate(codes) if '"scripts/smoke_test_lab.py"' in code)
    full = next(index for index, code in enumerate(codes) if 'OUTPUT = WORK / "ket_qua_lab"' in code)
    assert sample < full
    assert '"--frames", "20"' in codes[sample]
    assert codes[full].startswith('assert (SMOKE_ROOT / "DAT_KIEM_TRA.json").is_file()')
