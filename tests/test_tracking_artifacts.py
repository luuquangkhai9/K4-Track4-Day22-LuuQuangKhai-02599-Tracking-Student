"""Kiểm tra cache và metric bằng dữ liệu giả, không dùng GPU hoặc mạng."""

import json
from pathlib import Path

import pytest

from tracking_artifacts import DetectionCache, best_practice_case, detection_metadata, parse_summary


def test_metadata_changes_with_detector_and_images(tmp_path: Path) -> None:
    (tmp_path / "000001.jpg").write_bytes(b"anh gia")
    original = detection_metadata(tmp_path, 0.3, 0.5)
    assert original["frames"] == 1
    assert original["imgsz"] == 640
    assert original["class_id"] == 0
    assert original != detection_metadata(tmp_path, 0.15, 0.5)
    assert original != detection_metadata(tmp_path, 0.3, 0.7)
    (tmp_path / "000002.jpg").write_bytes(b"anh gia khac")
    assert original != detection_metadata(tmp_path, 0.3, 0.5)
    assert detection_metadata(tmp_path, 0.3, 0.5, 1) == original


def test_metadata_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        detection_metadata(tmp_path, 0.3, 0.5)


def test_cache_round_trip_keeps_empty_frames(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    metadata = {"frames": 2, "conf": 0.3}
    cache = DetectionCache(path, metadata)
    boxes = [[1.0, 2.0, 3.0, 4.0, 0.8, 0]]
    cache.frame(0, boxes)
    cache.frame(1, [])
    cache.close(True)
    reused = DetectionCache(path, metadata)
    assert reused.hit
    assert reused.frame(0) == boxes
    assert reused.frame(1) == []
    reused.close(True)


def test_failed_or_incomplete_cache_is_not_committed(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = DetectionCache(path, {"frames": 2})
    cache.frame(0, [])
    with pytest.raises(ValueError):
        cache.close(True)
    assert not path.exists()
    assert not cache.temporary.exists()
    cache = DetectionCache(path, {"frames": 1})
    cache.frame(0, [])
    cache.close(False)
    assert not path.exists()


def test_cache_rejects_wrong_configuration_and_order(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"frames": 1, "conf": 0.3}) + "\n" +
                    json.dumps({"frame": 1, "detections": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không khớp"):
        DetectionCache(path, {"frames": 1, "conf": 0.15})
    cache = DetectionCache(path, {"frames": 1, "conf": 0.3})
    try:
        with pytest.raises(ValueError, match="Sai thứ tự"):
            cache.frame(0)
    finally:
        cache.close(False)


def test_cache_rejects_trailing_records(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = DetectionCache(path, {"frames": 1})
    cache.frame(0, [])
    cache.close(True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{}\n')
    cache = DetectionCache(path, {"frames": 1})
    cache.frame(0)
    with pytest.raises(ValueError, match="thừa"):
        cache.close(True)


def test_summary_uses_header_names_and_allows_negative_mota() -> None:
    assert parse_summary("IDF1 DetA MOTA HOTA\n75.5 60 -2.5 65.2\n") == {
        "HOTA": 65.2, "MOTA": -2.5, "IDF1": 75.5,
    }


@pytest.mark.parametrize("text", ["", "HOTA MOTA\n1 2", "HOTA MOTA IDF1\n1 2",
                                  "HOTA MOTA IDF1\nnan 2 3", "HOTA MOTA IDF1\n1 inf 3"])
def test_summary_rejects_missing_or_invalid_metrics(text: str) -> None:
    with pytest.raises(ValueError):
        parse_summary(text)


def test_best_case_prioritizes_hota_then_identity() -> None:
    metrics = {
        "a": {"HOTA": 60, "MOTA": 90, "IDF1": 70},
        "b": {"HOTA": 61, "MOTA": 50, "IDF1": 70},
        "c": {"HOTA": 61, "MOTA": 40, "IDF1": 75},
    }
    assert best_practice_case(metrics) == "c"
    with pytest.raises(ValueError):
        best_practice_case({})


def test_best_case_tie_is_deterministic() -> None:
    metric = {"HOTA": 60, "MOTA": 80, "IDF1": 70}
    assert best_practice_case({"b": metric, "a": metric}) == "a"
