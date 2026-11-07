# G0-Kaggle smoke operation

Status: **ready for Kaggle verification**. Local CPU tests and lint passed. No GPU
run has been observed. G0-Kaggle remains pending; G0-Deployment on two RTX 5090s
is separately unverified. This task covers runtime preparation only.

Authoritative project instructions: [AGENTS.md](../AGENTS.md) and
[master plan v2](datamate-agent-plan-v2.md), especially sections 2, 5 and 6.
`architecture.md` is historical and was not used for implementation decisions.
The G0 audit confirmed the configured serving profile, manual offline assets,
package/notebook separation and separate Kaggle/deployment acceptance gates
against those authoritative instructions; no runtime correction was needed.

The observable result is one completed, nonempty Vietnamese normalization response
from local Qwen assets, with a recorded finish reason, actual token counts, exact
checkpoint/tokenizer/template revision, serving settings, dependency versions,
GPU/driver/CUDA details and elapsed time including preparation and loading. This
is a runtime smoke test; it does not establish answer accuracy or throughput.

## Candidate profile

`configs/kaggle_smoke.json` contains the operating settings and budgets: localhost
vLLM, text-only loading, two T4s with tensor parallelism, FP16, one sequence,
16,384-token context, thinking disabled, and one synthetic text request. There
is no model/precision/hardware fallback, prompt truncation or inference retry.
Output ending because of the token limit is a failure.

The checkpoint, tokenizer and chat template all use revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
The [publisher snapshot](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
and its [metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-9B/revision/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
support the pinned file inventory and full checkpoint parameter count. The
manifest retains license and publication evidence. Text-only deployed parameter
accounting remains pending; the publisher's full checkpoint count is not claimed
as a measurement of parameters actually loaded by vLLM.

The candidate stack is vLLM 0.17.1, PyTorch 2.10.0 and Transformers 4.57.6. vLLM's
[tagged implementation](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/model_executor/models/qwen3_5.py)
contains Qwen3.5 support; the
[serving recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
documents text-only loading and thinking controls. Kernel, driver and T4
compatibility still require this experiment. Stop and bring back the report if
the candidate fails; do not switch precision, checkpoint or hardware silently.

## Prepare assets while Internet is available

1. In the target Kaggle notebook, note `sys.version`, `platform.machine()` and
   `platform.libc_ver()`. Prepare wheels in a separate online Linux environment
   with the same Python minor version and compatible Linux/glibc. The smoke
   module uses only Python 3.11–3.13 standard-library features. The existing uv
   development project keeps its Python >=3.13 requirement; the notebook imports
   the smoke module directly from `src/`, without installing the unrelated data
   analysis dependencies into Kaggle.
2. Download the full model snapshot at the exact revision, including real
   safetensors weights, tokenizer files, `chat_template.jinja`, `config.json`,
   weight index and `LICENSE`. For example, using the Hugging Face CLI in your
   online preparation environment:

   ```bash
   hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir model
   ```

   Upload `model/` as a private Kaggle Dataset, preserving file bytes. A Git clone
   containing LFS pointer files is insufficient. Metadata hashes and tokenizer
   SHA256 are checked; weight shard sizes catch missing/partial files. Full
   weight SHA256 verification is deferred to release packaging; expected hashes
   are retained in `configs/qwen35-9b-assets.json` for that work.
3. In a clean, online Linux environment matching the target Python, resolve the
   candidate serving stack and save every dependency version and binary wheel:

   ```bash
   python -m venv prep-env
   prep-env/bin/python -m pip install -r configs/requirements-kaggle.in
   mkdir -p wheelhouse
   prep-env/bin/python -m pip freeze > wheelhouse/requirements-linux.lock
   prep-env/bin/python -m pip download --only-binary=:all: -r wheelhouse/requirements-linux.lock -d wheelhouse
   ```

   This is online asset preparation, not inference. If resolution, download or
   binary compatibility fails, retain that log; do not substitute an unpinned
   stack. Do not use Windows wheels, editable installs, URL requirements or a
   Windows `uv.lock` export for this Linux environment. The full Linux lock will
   be created by this step; it is not yet a GPU-validated release lock.
4. Verify offline installation in a second clean environment before uploading:

   ```bash
   python -m venv verify-env
   verify-env/bin/python -m pip --isolated install --no-index --only-binary=:all: --find-links wheelhouse -r wheelhouse/requirements-linux.lock
   verify-env/bin/python -m pip check
   ```

   Upload the complete `wheelhouse/`, including its lock, as another private
   Kaggle Dataset. PyTorch and CUDA dependencies make this asset large; check
   available disk space in both preparation and Kaggle environments.
5. Upload an application folder named `app/` containing only
   `src/zalo_aic_data/`, `configs/`, `prompts/`, and optionally these instructions.
   Import `notebooks/kaggle_smoke.ipynb` into Kaggle separately. Do not upload
   `.venv`, `train`, `public_test`, evaluation answers or the entire data repo.

## Run on Kaggle

1. Select **GPU T4 ×2** and set **Internet off** in notebook settings. Attach the
   three private assets: application, pinned model, Linux wheelhouse.
2. Edit only `APP`, `MODEL` and `WHEELS` in the notebook to match the actual Input
   folders. `MODEL` must directly contain `config.json`; `WHEELS` must directly
   contain `requirements-linux.lock` and the wheels.
3. Confirm Internet is disabled before leaving `offline_confirmed=True`. This
   records an operator confirmation, not an automated proof of Kaggle's network
   policy. Offline package/model environment flags are also enforced; serving
   and the single HTTP request use loopback only.
4. Run both code cells. A clean private environment is built inside the run's
   output folder, exclusively from the attached wheels. The runtime checks GPU
   identity, visible CUDA devices and an actual CUDA operation on each device,
   then launches vLLM. Preparation, installation and model loading consume the
   configured run budget. Cleanup has its own bounded shutdown allowance.
5. Download the evidence folder printed by the notebook. Keep the successful
   `report.json`, notebook output/settings and server log for gate review. Only
   after observing that offline two-T4 success should G0-Kaggle be marked passed.

The equivalent source-package CLI is:

```bash
PYTHONPATH=/kaggle/input/datamate-smoke/app/src python -m zalo_aic_data.kaggle_smoke --config /kaggle/input/datamate-smoke/app/configs/kaggle_smoke.json --model-path /kaggle/input/qwen35-9b/model --wheelhouse-path /kaggle/input/datamate-runtime/wheelhouse --output-root /kaggle/working/out/kaggle-smoke --offline-confirmed
```

## Expected output and recovery

The notebook prints the unique evidence folder, `Success: True | Stage:
completed`, and a Vietnamese rewrite of the synthetic business question. Wording
may vary; it should retain the named company, KH001, March 2026, the comparison
and the exclusion of cancelled orders. No revenue calculation is requested.
`report.json` contains `success`, `stage`, timings, the generated text, raw server
response, token usage, manifest, effective config, exact command, dependencies
and hardware. A completed response is not a semantic accuracy evaluation.

On failure, the notebook prints `Success: False`, the failing stage and error,
then raises an assertion so the run visibly fails. The CLI exits nonzero.
The report is written before expensive stages and finalized on handled errors.
If Kaggle kills the process, the last saved report may describe an in-progress
stage; a hard termination cannot guarantee final logs.

Bring back these small files, **not the generated `env/` folder**:

- `report.json` and `traceback.log` (when present).
- `bootstrap.log` for missing wheels, incompatible wheels or installation errors.
- `server.log` for CUDA/kernel, memory, model-loading and generation errors.
- `nvidia-smi.txt` and the copied `requirements-linux.lock`, when present.
- Kaggle cell output and a screenshot or description of accelerator and Internet
  settings. For asset preparation failures, include the online pip/hf error log
  and the target Python/Linux details.

Missing model files stop before any installation or model import, with the
expected local path and instructions to attach the complete asset. Missing
wheelhouse/lock files likewise have no download fallback. For a profile failure,
retain the original evidence before discussing a new candidate. This smoke test
does not emit competition answers, `unanswerable`, or evaluation scores.

## Local verification

```powershell
uv run pytest tests/test_kaggle_smoke.py
uv run ruff check .
uv run python -m zalo_aic_data.kaggle_smoke --model-path missing-model --output-root smoke-out
```

The last command should exit 1 and save an actionable missing-asset report.
CPU-only tests use tiny synthetic asset bytes and fake runtime responses; they
verify orchestration and failure handling, not GPU compatibility. The existing
data-inspection notebook is preserved, with only its pre-existing unused-import
and import-order lint findings exempted in `pyproject.toml`.

Recorded local evidence (2026-11-07, Windows, Python 3.13.7):

- `uv run python -m pytest tests/test_kaggle_smoke.py`: **51 passed**.
  `uv run pytest` was attempted first; Windows Application Control blocked its
  executable launcher. Invoking pytest through the project interpreter passed.
- `uv run ruff check .`: **passed**.
- The real CLI with `--model-path missing-model` exited **1**, recorded stage
  `assets`, and saved an actionable missing-`config.json` error and traceback in
  `smoke-out/52816ed7259249569f0ea5f872b9f74a/` before creating a GPU environment.
- Linux wheel installation, GPU loading/generation and the text-only deployed
  parameter count remain unverified. No G0 gate has been marked passed.
