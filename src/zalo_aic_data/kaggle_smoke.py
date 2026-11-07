"""One offline, synthetic request against the initial G0-Kaggle profile.

Only the standard library is imported here. GPU dependencies live in a separate
Linux environment populated exclusively from the manually attached wheelhouse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any


class SmokeError(RuntimeError):
    """An operator-actionable startup or generation failure."""


@dataclass(frozen=True)
class SmokeConfig:
    model_id: str
    revision: str
    model_path: str
    wheelhouse_path: str
    prompt_path: str
    asset_manifest_path: str
    runtime_requirements_path: str
    host: str
    port: int
    gpu_name: str
    gpu_count: int
    tensor_parallel_size: int
    dtype: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    enable_thinking: bool
    max_tokens: int
    temperature: float
    seed: int
    run_timeout_seconds: float
    request_timeout_seconds: float
    probe_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_timeout_seconds: float


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> SmokeConfig:
    raw = read_json(path)
    expected = {field.name for field in fields(SmokeConfig)}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise SmokeError(
            f"Config must contain exactly these settings: {sorted(expected)}"
        )
    config = SmokeConfig(**raw)
    for key, value in raw.items():
        if key == "enable_thinking":
            if value is not False:
                raise SmokeError(
                    "The initial smoke test requires enable_thinking=false."
                )
        elif key in {
            "port",
            "gpu_count",
            "tensor_parallel_size",
            "max_model_len",
            "max_num_seqs",
            "max_tokens",
            "seed",
        }:
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise SmokeError(
                    f"{key} must be a positive integer (seed may be zero)."
                )
        elif key.endswith("_seconds") or key in {
            "gpu_memory_utilization",
            "temperature",
        }:
            if (
                type(value) not in {float, int}
                or not math.isfinite(value)
                or value < 0
                or (key != "temperature" and value == 0)
            ):
                raise SmokeError(
                    f"{key} must be finite and positive (temperature may be zero)."
                )
        elif not isinstance(value, str) or not value.strip():
            raise SmokeError(f"{key} must be a nonempty string.")
    if config.model_id != "Qwen/Qwen3.5-9B" or not re.fullmatch(
        r"[a-f0-9]{40}", config.revision
    ):
        raise SmokeError("Use Qwen/Qwen3.5-9B with an exact checkpoint commit SHA.")
    if config.host != "127.0.0.1" or config.port > 65535:
        raise SmokeError("Serving must bind to 127.0.0.1 with a valid TCP port.")
    if config.dtype != "half" or config.gpu_name != "T4":
        raise SmokeError("G0-Kaggle requires FP16 (half) and T4 hardware.")
    if config.tensor_parallel_size != config.gpu_count:
        raise SmokeError("Tensor parallel size must match the required GPU count.")
    if not 0 < config.gpu_memory_utilization <= 1:
        raise SmokeError("gpu_memory_utilization must be in (0, 1].")
    if config.max_tokens >= config.max_model_len:
        raise SmokeError("Output budget must leave room for prompt tokens.")
    return config


def require_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SmokeError(
            f"Missing or empty {description}: {path}. Attach the complete offline asset "
            "and check its mount path; no download fallback is enabled. See docs/kaggle.md."
        )


def validate_assets(
    model: Path, manifest: dict[str, Any], config: SmokeConfig
) -> dict[str, Any]:
    for key in ("revision", "tokenizer_revision", "chat_template_revision"):
        if manifest.get(key) != config.revision:
            raise SmokeError(
                f"Asset manifest {key} does not match the configured checkpoint."
            )
    if manifest.get("model_id") != config.model_id:
        raise SmokeError("Asset manifest model_id does not match the configured model.")
    for name, expected in manifest["git_blobs"].items():
        path = model / name
        require_file(path, "model/tokenizer metadata")
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected:
            raise SmokeError(
                f"Checkpoint metadata mismatch: {path}. Reattach the pinned revision."
            )
    for name, expected in manifest["large_files"].items():
        path = model / name
        require_file(path, "model/tokenizer asset")
        if path.stat().st_size != expected["size"]:
            raise SmokeError(
                f"Incomplete asset: {path}. Reattach the full file, not a Git LFS pointer."
            )
    # Full multi-GB weight hashing belongs to later release packaging. The small
    # tokenizer is checked fully here; the report explicitly records the boundary.
    if (
        sha256(model / "tokenizer.json")
        != manifest["large_files"]["tokenizer.json"]["sha256"]
    ):
        raise SmokeError("Tokenizer hash mismatch. Reattach the pinned tokenizer.json.")
    model_config = read_json(model / "config.json")
    if model_config.get("quantization_config") or model_config.get(
        "text_config", {}
    ).get("quantization_config"):
        raise SmokeError("Quantized assets are outside the FP16 smoke profile.")
    index = read_json(model / "model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    expected_shards = {
        name for name in manifest["large_files"] if name.endswith(".safetensors")
    }
    if not shards or shards != expected_shards:
        raise SmokeError("Weight index does not match the pinned shard inventory.")
    return {
        "metadata_git_hashes": "verified",
        "tokenizer_sha256": "verified",
        "weight_sizes": "verified",
        "weight_sha256": "not checked in G0 smoke",
    }


def read_pins(path: Path) -> dict[str, str]:
    require_file(path, "fully pinned Linux requirements")
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise SmokeError(
                f"Unpinned/unsupported requirement in {path}: {line!r}. Use pip freeze."
            )
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in pins:
            raise SmokeError(f"Duplicate requirement: {name}")
        pins[name] = version
    if not pins:
        raise SmokeError(f"No pinned requirements in {path}.")
    return pins


def offline_environment() -> dict[str, str]:
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "PIP_NO_INDEX": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def server_command(config: SmokeConfig, python: Path, model: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--tokenizer",
        str(model),
        "--chat-template",
        str(model / "chat_template.jinja"),
        "--served-model-name",
        config.model_id,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--language-model-only",
        "--generation-config",
        "vllm",
        "--seed",
        str(config.seed),
    ]


def request_body(config: SmokeConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "model": config.model_id,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "seed": config.seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }


def parse_generation(raw: dict[str, Any]) -> Generation:
    choice = raw["choices"][0]
    text = choice["message"].get("content")
    if choice["finish_reason"] != "stop":
        raise SmokeError(
            f"Incomplete generation: finish_reason={choice['finish_reason']!r}."
        )
    if not isinstance(text, str) or not text.strip():
        raise SmokeError("Model returned no usable text.")
    if choice["message"].get("reasoning_content") or "<think>" in text:
        raise SmokeError("Unexpected thinking output with enable_thinking=false.")
    usage = raw["usage"]
    if any(
        type(usage.get(key)) is not int or usage[key] <= 0
        for key in ("prompt_tokens", "completion_tokens")
    ):
        raise SmokeError("Server did not return positive actual token counts.")
    return Generation(
        text,
        choice["finish_reason"],
        usage["prompt_tokens"],
        usage["completion_tokens"],
    )


class VllmRuntime:
    """Own the isolated serving environment, localhost process and deadline."""

    def __init__(self, config: SmokeConfig, run_dir: Path, started: float):
        self.config = config
        self.run_dir = run_dir
        self.deadline = started + config.run_timeout_seconds
        self.env = offline_environment()
        self.python = run_dir / "env" / "bin" / "python"
        self.process: subprocess.Popen | None = None
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def remaining(self, cap: float | None = None) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise SmokeError(
                "Run deadline exhausted, including preparation and model loading."
            )
        return min(remaining, cap) if cap is not None else remaining

    def command(self, args: list[str], *, log_path: Path | None = None) -> str:
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    args,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self.env,
                    timeout=self.remaining(),
                    check=False,
                )
            output = f"See {log_path}"
        else:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                env=self.env,
                timeout=self.remaining(self.config.probe_timeout_seconds),
                check=False,
            )
            output = result.stdout
        if result.returncode:
            detail = output if log_path else f"{result.stdout}\n{result.stderr}"
            raise SmokeError(
                f"Command failed ({args[0]}, exit {result.returncode}). {detail}"
            )
        return output

    def prepare(self, wheels: Path, lock: Path) -> dict[str, str]:
        environment = str(self.python.parent.parent)
        create = [sys.executable, "-m", "venv"]
        pip = [str(self.python), "-m", "pip"]
        # Kaggle's base pip can manage the venv even if ensurepip is broken.
        if find_spec("pip") is not None:
            create.append("--without-pip")
            pip = [sys.executable, "-m", "pip", "--python", environment]
        pip += ["--isolated", "--disable-pip-version-check"]
        bootstrap_log = self.run_dir / "bootstrap.log"
        self.command([*create, environment], log_path=bootstrap_log)
        self.command(
            [
                *pip,
                "install",
                "--no-index",
                "--only-binary=:all:",
                "--find-links",
                str(wheels),
                "--requirement",
                str(lock),
            ],
            log_path=bootstrap_log,
        )
        installed = json.loads(
            self.command(
                [
                    str(self.python),
                    "-c",
                    (
                        "import importlib.metadata,json,re; "
                        "print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version "
                        "for d in importlib.metadata.distributions()}))"
                    ),
                ]
            )
        )
        for name, version in read_pins(lock).items():
            if installed.get(name) != version:
                raise SmokeError(
                    f"Installed version mismatch: {name} expected {version}."
                )
        unpinned = set(installed) - set(read_pins(lock)) - {"pip", "setuptools"}
        if unpinned:
            raise SmokeError(
                f"Linux lock omitted installed dependencies: {sorted(unpinned)}."
            )
        self.command([*pip, "check"], log_path=bootstrap_log)
        return installed

    def hardware(self) -> dict[str, Any]:
        smi = self.command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        (self.run_dir / "nvidia-smi.txt").write_text(smi, encoding="utf-8")
        devices = list(csv.reader(smi.strip().splitlines(), skipinitialspace=True))
        if len(devices) != self.config.gpu_count or any(
            self.config.gpu_name not in row[1] for row in devices
        ):
            raise SmokeError(
                f"Select {self.config.gpu_count} T4 GPUs in Kaggle. Observed: {smi.strip()}"
            )
        probe = self.command(
            [
                str(self.python),
                "-c",
                (
                    "import json,torch; "
                    "gpus=[{'name':torch.cuda.get_device_name(i),"
                    "'capability':torch.cuda.get_device_capability(i)} "
                    "for i in range(torch.cuda.device_count())]; "
                    "[(torch.ones(1,device=f'cuda:{i}')+1).cpu() for i in range(len(gpus))]; "
                    "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':gpus}))"
                ),
            ]
        )
        result = json.loads(probe)
        if len(result["gpus"]) != self.config.gpu_count or any(
            self.config.gpu_name not in gpu["name"] for gpu in result["gpus"]
        ):
            raise SmokeError(
                f"CUDA-visible hardware differs from the requested profile: {result}"
            )
        return {
            "nvidia_smi": devices,
            "cuda_probe": result,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def exchange(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.config.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with self.http.open(request, timeout=self.remaining(timeout)) as response:
            content = response.read()
        return json.loads(content) if content else {}

    def start(self, command: list[str]) -> None:
        if sys.platform != "linux":
            raise SmokeError("vLLM serving requires the Linux Kaggle environment.")
        with socket.socket() as probe:
            try:
                probe.bind((self.config.host, self.config.port))
            except OSError as exc:
                raise SmokeError(
                    "Local serving port is occupied. Stop the previous smoke run."
                ) from exc
        with (self.run_dir / "server.log").open("w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,
            )
        while True:
            self.remaining()
            if self.process.poll() is not None:
                raise SmokeError(
                    "vLLM exited during loading. Read server.log; no profile fallback was attempted."
                )
            try:
                self.exchange("/health", timeout=self.config.poll_interval_seconds)
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(self.remaining(self.config.poll_interval_seconds))

    def close(self) -> None:
        if sys.platform != "linux" or self.process is None:
            return
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=self.config.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=self.config.shutdown_timeout_seconds)


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_smoke(
    config_path: Path,
    *,
    model_path: Path | None = None,
    wheelhouse_path: Path | None = None,
    output_root: Path = Path("smoke-out"),
    offline_confirmed: bool = False,
) -> dict[str, Any]:
    """Save evidence on every handled failure; never claim a gate pass here."""
    started = time.monotonic()
    run_dir = output_root.resolve() / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "success": False,
        "stage": "configuration",
        "error": None,
        "started_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "environment": {"platform": platform.platform(), "python": sys.version},
        "offline_confirmation": offline_confirmed,
        "generated_response": None,
        "gate_status": "G0-Kaggle pending review of actual offline T4 evidence",
        "deployment_status": "G0-Deployment unverified",
    }
    runtime = None
    try:
        config_path = config_path.resolve()
        root = config_path.parent.parent
        config = load_config(config_path)
        config = replace(
            config,
            model_path=str(model_path or config.model_path),
            wheelhouse_path=str(wheelhouse_path or config.wheelhouse_path),
        )
        report["config"] = asdict(config)
        report["config_sha256"] = sha256(config_path)
        write_report(run_dir / "report.json", report)
        report["stage"] = "assets"
        manifest_path = root / config.asset_manifest_path
        require_file(manifest_path, "checkpoint manifest in the application asset")
        manifest = read_json(manifest_path)
        report["model_manifest"] = manifest
        report["asset_checks"] = validate_assets(
            Path(config.model_path), manifest, config
        )
        messages_path = root / config.prompt_path
        require_file(messages_path, "versioned smoke prompt in the application asset")
        messages = read_json(messages_path)
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or set(m) != {"role", "content"}
                or m["role"] not in {"system", "user"}
                or not isinstance(m["content"], str)
                or not m["content"].strip()
                for m in messages
            )
        ):
            raise SmokeError(
                "Smoke prompt must contain nonempty system/user text messages."
            )
        report["prompt_sha256"] = sha256(messages_path)
        report["request"] = request_body(config, messages)
        wheels = Path(config.wheelhouse_path)
        lock = wheels / "requirements-linux.lock"
        pins = read_pins(lock)
        for name, version in read_pins(root / config.runtime_requirements_path).items():
            if pins.get(name) != version:
                raise SmokeError(
                    f"Wheelhouse lock must pin {name}=={version}. Rebuild the offline assets."
                )
        if not any(wheels.glob("*.whl")):
            raise SmokeError(
                f"No Linux wheels in {wheels}. Attach the complete offline wheelhouse."
            )
        report["runtime_lock_sha256"] = sha256(lock)
        (run_dir / "requirements-linux.lock").write_bytes(lock.read_bytes())
        if not offline_confirmed:
            raise SmokeError(
                "Disable Kaggle Internet, then pass offline_confirmed=True / --offline-confirmed."
            )
        if sys.platform != "linux":
            raise SmokeError(
                "GPU execution requires Linux Kaggle with two T4s; local asset checks completed."
            )
        runtime = VllmRuntime(config, run_dir, started)
        report["stage"] = "offline_install"
        write_report(run_dir / "report.json", report)
        report["dependencies"] = runtime.prepare(wheels, lock)
        report["stage"] = "hardware"
        report["hardware"] = runtime.hardware()
        command = server_command(config, runtime.python, Path(config.model_path))
        report["server_command"] = command
        report["offline_environment"] = {
            k: v
            for k, v in runtime.env.items()
            if k not in os.environ or os.environ[k] != v
        }
        report["stage"] = "model_loading"
        write_report(run_dir / "report.json", report)
        load_started = time.monotonic()
        runtime.start(command)
        report["model_load_seconds"] = time.monotonic() - load_started
        report["stage"] = "generation"
        write_report(run_dir / "report.json", report)
        generation_started = time.monotonic()
        raw = runtime.exchange(
            "/v1/chat/completions",
            report["request"],
            timeout=config.request_timeout_seconds,
        )
        report["generation_seconds"] = time.monotonic() - generation_started
        report["raw_response"] = raw
        report["generated_response"] = (
            raw.get("choices", [{}])[0].get("message", {}).get("content")
        )
        report["generation"] = asdict(parse_generation(raw))
        runtime.remaining()
        report["success"] = True
        report["stage"] = "completed"
    except Exception as exc:  # noqa: BLE001 - persist the top-level failure evidence
        report["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:  # noqa: BLE001 - preserve evidence even if cleanup fails
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                report["success"] = False
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(run_dir / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/kaggle_smoke.json")
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--wheelhouse-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("smoke-out"))
    parser.add_argument("--offline-confirmed", action="store_true")
    args = parser.parse_args()
    report = run_smoke(
        args.config,
        model_path=args.model_path,
        wheelhouse_path=args.wheelhouse_path,
        output_root=args.output_root,
        offline_confirmed=args.offline_confirmed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
