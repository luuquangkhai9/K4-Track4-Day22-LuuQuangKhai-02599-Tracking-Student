"""Kiểm tra cell cài môi trường trên kernel giả Python 3.13, không tải thư viện."""

import json
import os
import types
from pathlib import Path
import run_kaggle_lab


def test_python313_kernel_installs_isolated_python311(tmp_path: Path, monkeypatch) -> None:
    notebook = json.loads((Path(__file__).resolve().parents[1] / "kaggle_run_all.ipynb")
                          .read_text(encoding="utf-8"))
    setup = next("".join(cell["source"]) for cell in notebook["cells"]
                 if cell["cell_type"] == "code" and "UV_TOOLS =" in "".join(cell["source"]))
    calls = []

    def fake_logged(command, log_path, env_overrides=None):
        assert log_path.is_relative_to(tmp_path / "working/logs_cai_dat")
        calls.append((command, {"env": env_overrides}))

    monkeypatch.setattr(run_kaggle_lab, "run_logged", fake_logged)

    context = {
        "sys": types.SimpleNamespace(executable="/usr/bin/python313", version="3.13.15",
                                     version_info=(3, 13, 15)),
        "os": os,
        "TEMP": tmp_path, "REPO": tmp_path / "repo", "TRACKEVAL_COMMIT": "commit-gia",
        "WORK": tmp_path / "working",
        "shutil": types.SimpleNamespace(which=lambda name: "/usr/bin/ffmpeg"),
    }
    exec(compile(setup, "cell-cai-moi-truong", "exec"), context)
    commands = [command for command, _ in calls]
    assert commands[0] == ["nvidia-smi", "-L"]
    install_python = next(command for command in commands if command[1:3] == ["-m", "uv"]
                          and command[3:5] == ["python", "install"])
    assert install_python[-1] == "3.11"
    venv = next(command for command in commands if "--seed" in command)
    assert "--system-site-packages" not in venv
    assert venv[venv.index("--python") + 1] == "3.11"
    child = str(tmp_path / "venv_py311/bin/python")
    assert str(context["PYTHON"]) == child
    torch = next(command for command in commands if "torch==2.6.0" in command)
    assert torch[0] == child and "torchvision==0.21.0" in torch
    assert torch[-1] == "https://download.pytorch.org/whl/cu124"
    dependencies = next(command for command in commands if "-r" in command)
    assert dependencies[0] == child
    uv_envs = [kwargs["env"] for command, kwargs in calls if command[1:3] == ["-m", "uv"]]
    assert all(env["PYTHONPATH"] == str(tmp_path / "uv_tools") for env in uv_envs)
    assert all(env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "pythons") for env in uv_envs)
