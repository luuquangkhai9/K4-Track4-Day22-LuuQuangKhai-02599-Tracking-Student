"""Kiểm tra kết nối device và cache với mô hình giả, không tải trọng số."""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


class FakeArray(list):
    """Mảng giả đủ giao diện dùng khi trao đổi detection với cache."""

    def tolist(self):
        """Trả các hộp để ghi JSONL."""
        return list(self)

    def reshape(self, *shape):
        """Giữ nguyên dữ liệu giả đã có đúng số cột."""
        return self


def load_tracking(monkeypatch):
    """Nạp script với thư viện giả để kiểm tra mà không cài mô hình."""
    loads = []
    devices = []

    class FakeYOLO:
        def __init__(self, weights):
            loads.append(weights)

        def to(self, device):
            devices.append(device)
            return self

    numpy = types.ModuleType("numpy")
    numpy.asarray = lambda data, dtype: FakeArray(data)
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    torch = types.ModuleType("torch")
    torch.device = lambda value: types.SimpleNamespace(type=value.split(":")[0], value=value)
    zoo = types.ModuleType("boxmot.tracker_zoo")
    zoo.get_tracker_config = lambda name: name
    tracker_devices = []

    def create_tracker(**kwargs):
        tracker_devices.append(kwargs["device"])
        return types.SimpleNamespace(update=lambda detections, frame: [
            [1, 2, 11, 22, 5, 0.9, 0]
        ] if detections else [])

    zoo.create_tracker = create_tracker
    for name, module in {"numpy": numpy, "cv2": types.ModuleType("cv2"),
                         "torch": torch, "ultralytics": ultralytics,
                         "boxmot": types.ModuleType("boxmot"), "boxmot.tracker_zoo": zoo}.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "scripts/run_tracking.py"
    spec = importlib.util.spec_from_file_location("tracking_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, loads, devices, tracker_devices


def test_cached_run_skips_yolo_and_keeps_empty_frame(tmp_path, monkeypatch) -> None:
    module, loads, devices, tracker_devices = load_tracking(monkeypatch)
    images = tmp_path / "img1"
    images.mkdir()
    for index in (1, 2):
        (images / f"{index:06}.jpg").write_bytes(b"anh gia")
    module.iter_frames = lambda source: iter([(0, "co nguoi"), (1, "khong co nguoi")])
    detection_calls = []

    def detect(detector, frame, conf, iou):
        detection_calls.append(frame)
        return FakeArray([[1, 2, 11, 22, 0.9, 0]] if frame == "co nguoi" else [])

    module.detect = detect
    args = argparse.Namespace(source=str(images), seq_name="video_1", tracker="bytetrack",
                              conf=0.3, iou=0.5, device="cuda:0", save_video=False,
                              fps=20, max_frames=0, detection_cache=str(tmp_path / "cache.jsonl"))
    args.out = str(tmp_path / "first")
    module.run(args)
    args.tracker = "botsort"
    args.out = str(tmp_path / "second")
    module.run(args)
    assert loads == ["yolo26n.pt"]
    assert devices == ["cuda:0"]
    assert all(device.type == "cuda" for device in tracker_devices)
    assert len(detection_calls) == 2
    assert (tmp_path / "first/video_1.txt").read_text() == (tmp_path / "second/video_1.txt").read_text()
    summary = json.loads((tmp_path / "second/video_1_run.json").read_text())
    assert summary["frames_processed"] == 2 and summary["rows"] == 1
    assert summary["detection_cache_hit"] is True


def test_model_failure_does_not_commit_or_leave_cache(tmp_path, monkeypatch) -> None:
    module, _, _, _ = load_tracking(monkeypatch)
    images = tmp_path / "img1"
    images.mkdir()
    (images / "000001.jpg").write_bytes(b"anh gia")

    def fail(weights):
        raise RuntimeError("loi nap mo hinh gia")

    module.YOLO = fail
    cache = tmp_path / "cache.jsonl"
    args = argparse.Namespace(source=str(images), seq_name="video_1", tracker="bytetrack",
                              conf=0.3, iou=0.5, device="cpu", save_video=False, fps=20,
                              max_frames=0, detection_cache=str(cache), out=str(tmp_path / "output"))
    with pytest.raises(RuntimeError, match="nap mo hinh"):
        module.run(args)
    assert not cache.exists() and not cache.with_suffix(".jsonl.tmp").exists()
    assert not (tmp_path / "output/video_1_run.json").exists()
