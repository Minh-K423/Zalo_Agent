"""CPU-only behavioral checks. Fake model bytes/runtime are never gate evidence."""

import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from zalo_aic_data import kaggle_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kaggle_smoke.json"


@pytest.fixture
def config():
    return smoke.load_config(CONFIG)


@pytest.fixture
def assets(tmp_path, config):
    """Tiny self-consistent assets let us exercise checks without GPU weights."""
    model = tmp_path / "model"
    model.mkdir()
    small = {
        "config.json": json.dumps({"model_type": "qwen3_5"}),
        "tokenizer_config.json": "{}",
        "chat_template.jinja": "{{ messages }}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"weight": "part.safetensors"}}
        ),
        "LICENSE": "synthetic",
    }
    manifest = {
        "model_id": config.model_id,
        "revision": config.revision,
        "tokenizer_revision": config.revision,
        "chat_template_revision": config.revision,
        "git_blobs": {},
        "large_files": {},
    }
    for name, content in small.items():
        data = content.encode()
        (model / name).write_bytes(data)
        manifest["git_blobs"][name] = hashlib.sha1(
            f"blob {len(data)}\0".encode() + data
        ).hexdigest()
    for name in ("tokenizer.json", "part.safetensors"):
        data = b"synthetic fixture"
        (model / name).write_bytes(data)
        manifest["large_files"][name] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return model, manifest


@pytest.fixture
def app(tmp_path, assets):
    model, manifest = assets
    root = tmp_path / "app"
    (root / "configs").mkdir(parents=True)
    (root / "prompts").mkdir()
    config = root / "configs/kaggle_smoke.json"
    config.write_bytes(CONFIG.read_bytes())
    (root / "configs/qwen35-9b-assets.json").write_text(json.dumps(manifest))
    (root / "configs/requirements-kaggle.in").write_bytes(
        (ROOT / "configs/requirements-kaggle.in").read_bytes()
    )
    (root / "prompts/kaggle_smoke_v1.json").write_bytes(
        (ROOT / "prompts/kaggle_smoke_v1.json").read_bytes()
    )
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    (wheels / "requirements-linux.lock").write_bytes(
        (ROOT / "configs/requirements-kaggle.in").read_bytes()
    )
    (wheels / "fake.whl").write_bytes(b"test only")
    return {
        "config_path": config,
        "model_path": model,
        "wheelhouse_path": wheels,
        "output_root": tmp_path / "out",
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("host", "0.0.0.0"),
        ("port", 70000),
        ("port", True),
        ("max_tokens", "256"),
        ("max_tokens", 16384),
        ("max_model_len", 0),
        ("enable_thinking", True),
        ("enable_thinking", 0),
        ("dtype", "auto"),
        ("gpu_count", 1),
        ("gpu_name", "P100"),
        ("revision", "main"),
        ("gpu_memory_utilization", 1.1),
        ("run_timeout_seconds", -1),
        ("poll_interval_seconds", float("nan")),
        ("model_id", "another/model"),
    ],
)
def test_invalid_settings_fail_startup(tmp_path, key, value):
    raw = smoke.read_json(CONFIG)
    raw[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(smoke.SmokeError):
        smoke.load_config(path)


def test_unknown_setting_is_not_silently_ignored(tmp_path):
    raw = smoke.read_json(CONFIG)
    raw["typo"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(smoke.SmokeError, match="exactly"):
        smoke.load_config(path)


def test_pinned_profile_and_request(config):
    assert (config.gpu_count, config.tensor_parallel_size, config.max_num_seqs) == (
        2,
        2,
        1,
    )
    assert config.max_model_len == 16384
    command = smoke.server_command(config, Path("python"), Path("local-model"))
    assert command[command.index("--model") + 1] == "local-model"
    assert command[command.index("--tokenizer") + 1] == "local-model"
    assert "--language-model-only" in command
    assert "--trust-remote-code" not in command
    body = smoke.request_body(config, smoke.read_json(ROOT / config.prompt_path))
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "truncate_prompt_tokens" not in body
    assert "KH001" in body["messages"][1]["content"]
    assert smoke.offline_environment()["HF_HUB_OFFLINE"] == "1"
    assert smoke.offline_environment()["PIP_NO_INDEX"] == "1"


def test_valid_assets(config, assets):
    model, manifest = assets
    checks = smoke.validate_assets(model, manifest, config)
    assert checks["tokenizer_sha256"] == "verified"
    assert checks["weight_sha256"] == "not checked in G0 smoke"


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "part.safetensors"])
def test_missing_asset_is_actionable(config, assets, name):
    model, manifest = assets
    (model / name).unlink()
    with pytest.raises(smoke.SmokeError, match="Attach.*no download fallback"):
        smoke.validate_assets(model, manifest, config)


def test_incomplete_weight_file(config, assets):
    model, manifest = assets
    (model / "part.safetensors").write_bytes(b"partial")
    with pytest.raises(smoke.SmokeError, match="Incomplete asset"):
        smoke.validate_assets(model, manifest, config)


def test_metadata_mismatch(config, assets):
    model, manifest = assets
    (model / "config.json").write_text('{"quantization_config": {"bits": 4}}')
    with pytest.raises(smoke.SmokeError, match="metadata mismatch"):
        smoke.validate_assets(model, manifest, config)


def test_wrong_tokenizer_revision(config, assets):
    model, manifest = assets
    manifest["tokenizer_revision"] = "other"
    with pytest.raises(smoke.SmokeError, match="tokenizer_revision"):
        smoke.validate_assets(model, manifest, config)


@pytest.mark.parametrize(
    "requirement",
    [
        "vllm>=0.17",
        "https://example.com/pkg.whl",
        "--index-url https://example.com",
        "-e .",
        "pkg @ file:///x",
    ],
)
def test_lock_rejects_unpinned_or_network_sources(tmp_path, requirement):
    path = tmp_path / "requirements.lock"
    path.write_text(requirement)
    with pytest.raises(smoke.SmokeError, match="Unpinned"):
        smoke.read_pins(path)


def response(finish_reason="stop", text="Câu hỏi đã được chuẩn hóa."):
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


@pytest.mark.parametrize(
    ("finish", "text"),
    [("length", "partial"), ("stop", ""), ("stop", "<think>reasoning")],
)
def test_bad_generation_is_failure(finish, text):
    with pytest.raises(smoke.SmokeError):
        smoke.parse_generation(response(finish, text))


def test_generation_requires_actual_tokens():
    raw = response()
    raw["usage"]["prompt_tokens"] = None
    with pytest.raises(smoke.SmokeError, match="token counts"):
        smoke.parse_generation(raw)


def test_missing_assets_fail_before_install_or_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing assets must fail before resource creation")

    monkeypatch.setattr(smoke, "VllmRuntime", forbidden)
    result = smoke.run_smoke(
        CONFIG, model_path=tmp_path / "absent", output_root=tmp_path / "out"
    )
    assert not result["success"]
    assert result["stage"] == "assets"
    assert "no download fallback" in result["error"]
    assert (Path(result["run_dir"]) / "report.json").is_file()
    assert (Path(result["run_dir"]) / "traceback.log").is_file()


def test_offline_confirmation_required(app):
    result = smoke.run_smoke(**app)
    assert not result["success"]
    assert "Disable Kaggle Internet" in result["error"]


def test_wrong_runtime_lock_fails_before_install(app):
    (app["wheelhouse_path"] / "requirements-linux.lock").write_text("vllm==0.1\n")
    result = smoke.run_smoke(**app, offline_confirmed=True)
    assert "Wheelhouse lock must pin" in result["error"]


def test_missing_wheelhouse_is_actionable(app):
    app["wheelhouse_path"] = app["wheelhouse_path"] / "missing"
    result = smoke.run_smoke(**app, offline_confirmed=True)
    assert not result["success"]
    assert "requirements-linux.lock" in result["error"]
    assert "Attach" in result["error"]


@pytest.mark.parametrize(
    "smi",
    ["0, Tesla P100, uuid, driver, 16000\n", "0, Tesla T4, uuid, driver, 15000\n"],
)
def test_wrong_hardware_stops_before_model_loading(config, tmp_path, monkeypatch, smi):
    runtime = smoke.VllmRuntime(config, tmp_path, time.monotonic())
    monkeypatch.setattr(runtime, "command", lambda args: smi)
    with pytest.raises(smoke.SmokeError, match="Select 2 T4 GPUs"):
        runtime.hardware()
    assert (tmp_path / "nvidia-smi.txt").read_text() == smi


def test_cuda_visibility_mismatch_is_actionable(config, tmp_path, monkeypatch):
    runtime = smoke.VllmRuntime(config, tmp_path, time.monotonic())
    smi = "0, Tesla T4, uuid0, driver, 15000\n1, Tesla T4, uuid1, driver, 15000\n"
    results = iter([smi, json.dumps({"gpus": []})])
    monkeypatch.setattr(runtime, "command", lambda args: next(results))
    with pytest.raises(smoke.SmokeError, match="CUDA-visible hardware"):
        runtime.hardware()


@pytest.mark.parametrize(
    "failure", [None, "install", "hardware", "load", "generation", "truncated"]
)
def test_orchestration_records_outcome_and_closes_runtime(app, monkeypatch, failure):
    closed = []

    class FakeRuntime:
        def __init__(self, config, run_dir, started):
            self.python = Path("fake-python")
            self.env = smoke.offline_environment()

        def prepare(self, wheels, lock):
            if failure == "install":
                raise smoke.SmokeError("missing wheel; see bootstrap.log")
            return {"vllm": "test-double"}

        def hardware(self):
            if failure == "hardware":
                raise smoke.SmokeError("wrong GPU")
            return {"gpus": ["fake T4", "fake T4"]}

        def start(self, command):
            if failure == "load":
                raise smoke.SmokeError("model failed; see server.log")

        def exchange(self, path, body, timeout):
            if failure == "generation":
                raise TimeoutError("request timed out")
            return response("length" if failure == "truncated" else "stop")

        def remaining(self):
            return 1

        def close(self):
            closed.append(True)

    monkeypatch.setattr(smoke.sys, "platform", "linux")
    monkeypatch.setattr(smoke, "VllmRuntime", FakeRuntime)
    result = smoke.run_smoke(**app, offline_confirmed=True)
    assert result["success"] is (failure is None)
    assert closed == [True]
    saved = smoke.read_json(Path(result["run_dir"]) / "report.json")
    assert saved == result
    assert result["elapsed_seconds"] >= 0
    assert "pending" in result["gate_status"]
    assert result["deployment_status"].endswith("unverified")
    if failure == "truncated":
        assert result["generated_response"]
        assert result["raw_response"]["choices"][0]["finish_reason"] == "length"
    if failure is None:
        assert result["generation"]["prompt_tokens"] == 100
        assert result["model_load_seconds"] >= 0


def test_exhausted_budget_stops_subprocess(config, tmp_path):
    runtime = smoke.VllmRuntime(
        replace(config, run_timeout_seconds=1), tmp_path, time.monotonic() - 2
    )
    with pytest.raises(smoke.SmokeError, match="deadline exhausted"):
        runtime.command(["must-not-execute"])


def test_offline_bootstrap_uses_only_wheels(config, tmp_path, monkeypatch):
    runtime = smoke.VllmRuntime(config, tmp_path, time.monotonic())
    lock = ROOT / "configs/requirements-kaggle.in"
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return json.dumps(smoke.read_pins(lock)) if "-c" in args else ""

    monkeypatch.setattr(runtime, "command", command)
    runtime.prepare(tmp_path, lock)
    install = commands[1]
    assert "--isolated" in install
    assert "--no-index" in install
    assert "--only-binary=:all:" in install


def test_server_exit_is_reported(config, tmp_path, monkeypatch):
    runtime = smoke.VllmRuntime(replace(config, port=0), tmp_path, time.monotonic())

    class Exited:
        def poll(self):
            return 1

    def launch(*args, stdout, **kwargs):
        stdout.write("simulated loader failure\n")
        return Exited()

    monkeypatch.setattr(smoke.sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(smoke.SmokeError, match="server.log"):
        runtime.start(["test"])
    assert (tmp_path / "server.log").read_text(encoding="utf-8") == (
        "simulated loader failure\n"
    )


def test_shutdown_reaps_process_that_exits_before_forced_kill(
    config, tmp_path, monkeypatch
):
    runtime = smoke.VllmRuntime(config, tmp_path, time.monotonic())
    process = Mock(pid=123)
    timeout = config.shutdown_timeout_seconds
    process.wait.side_effect = [subprocess.TimeoutExpired("vllm", timeout), 0]
    killpg = Mock(side_effect=[None, ProcessLookupError()])

    monkeypatch.setattr(runtime, "process", process)
    monkeypatch.setattr(smoke.sys, "platform", "linux")
    monkeypatch.setattr(smoke.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(smoke.signal, "SIGKILL", 9, raising=False)

    runtime.close()

    assert killpg.call_args_list == [
        call(process.pid, smoke.signal.SIGTERM),
        call(process.pid, 9),
    ]
    assert process.wait.call_args_list == [call(timeout=timeout), call(timeout=timeout)]


def test_notebook_is_thin_and_python_compiles():
    nb = smoke.read_json(ROOT / "notebooks/kaggle_smoke.ipynb")
    cells = [
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    ]
    assert len(cells) == 2
    for source in cells:
        compile(source, "kaggle_smoke.ipynb", "exec")
    assert "run_smoke(" in cells[-1]
    assert "vllm" not in "".join(cells)
