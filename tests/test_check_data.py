"""Tests for check() without lab images or a network."""

from pathlib import Path

from check_data import check, describe_data_root, discover_data_roots, usable_data_root, main

import pytest


def _touch_jpg(directory: Path, name: str = "000001.jpg") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"")


def _lab_tree(root: Path, with_practice_gt: bool = True, extra_images: bool = True) -> None:
    for index in range(1, 6):
        name = f"video_{index}"
        if extra_images or name == "video_1":
            _touch_jpg(root / name / "img1")
        else:
            (root / name).mkdir(parents=True, exist_ok=True)
    if with_practice_gt:
        gt_dir = root / "video_1" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n", encoding="utf-8")


def test_ready_pack_passes(tmp_path: Path) -> None:
    _lab_tree(tmp_path)
    assert check(tmp_path) is True


def test_missing_images_fails(tmp_path: Path) -> None:
    _lab_tree(tmp_path, extra_images=False)
    assert check(tmp_path) is False


def test_video_without_labels_still_passes(tmp_path: Path) -> None:
    _lab_tree(tmp_path)
    assert not (tmp_path / "video_2" / "gt").exists()
    assert check(tmp_path) is True


def test_practice_video_without_labels_fails(tmp_path: Path) -> None:
    _lab_tree(tmp_path, with_practice_gt=False)
    assert check(tmp_path) is False


def test_discovery_ignores_empty_directories_and_finds_nested_pack(tmp_path: Path) -> None:
    empty = tmp_path / "data_lab21"
    for index in range(1, 6):
        (empty / f"video_{index}/img1").mkdir(parents=True)
    actual = tmp_path / "dataset/giai_nen/data_lab21"
    _lab_tree(actual)
    assert not usable_data_root(empty)
    assert usable_data_root(actual)
    assert discover_data_roots([tmp_path, actual.parent]) == [actual.resolve()]


def test_discovery_requires_labels_for_practice(tmp_path: Path) -> None:
    _lab_tree(tmp_path, with_practice_gt=False)
    assert discover_data_roots([tmp_path]) == []


def test_description_shows_files_inside_img1(tmp_path: Path) -> None:
    _lab_tree(tmp_path)
    (tmp_path / "video_1/img1/images.zip").write_bytes(b"zip gia")
    description = describe_data_root(tmp_path)
    assert description["root_exists"]
    assert description["videos"]["video_1"]["jpg_files"] == 1
    assert description["videos"]["video_1"]["gt_exists"]
    assert "images.zip" in description["videos"]["video_1"]["sample_entries"]
    missing = describe_data_root(tmp_path / "duong_dan_sai")
    assert not missing["root_exists"]
    assert not missing["videos"]["video_1"]["img1_exists"]


def test_cli_fails_immediately_when_data_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["check_data.py", "--lab-data-root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
