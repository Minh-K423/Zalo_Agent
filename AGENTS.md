# DataMate project brief

Build an offline batch solver for Vietnamese business questions over tables and documents. Reuse the pinned `Qwen/Qwen3.5-9B` checkpoint for query interpretation and SQL generation. Develop on Kaggle and ship an offline Docker entry point. The competition deadline includes model loading and data preparation. The parameter ceiling applies to all models used at inference. Keep numeric budgets, limits and thresholds in `configs/`. See [competition constraints](rules.md) and the [master plan](docs/datamate-agent-plan-v2.md).

Submission types are `number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, `unanswerable` and `abstain`. Use `unanswerable` for a supported finding that the required information is absent. Use `abstain` for unresolved questions or processing failures.

## Current gate and work style

**G0-Kaggle.** Done when the pinned model loads from local assets in an offline Kaggle T4 session and completes a representative request. Record the model revision, serving configuration, hardware and outcome. Track the deployment hardware check separately (§2, §6).

Take one small task per session within the active gate. Define its observable result, test the behavior and relevant failure boundary, and report what passed or remains unverified. Update the current gate when its acceptance evidence is recorded.

## Repository and commands

Use the existing uv project at the repo root. Add modules under `src/zalo_aic_data/` as each gate needs them. Map the package responsibilities in §5 to that directory. Put tests in `tests/`, versioned prompts in `prompts/` and operating settings in `configs/`. Keep notebooks thin and put reusable logic in the package.

Use pure functions for parsing, normalization, context assembly and formatting. Use typed records at boundaries and classes for resources such as database sessions and the model runtime. Validate config at startup and pass it explicitly.

From the repo root, once the development dependencies are configured:

- Run the tests with `uv run pytest`. For a scoped change, pass the affected test file.
- Run lint with `uv run ruff check .`.

**Ask before** adding dependencies, changing the repo structure, or starting inference-side work that would touch evaluation code or gold answers. Keep scoring in a separate evaluator.

## Hard rules (verbatim from the master plan)

- One allowed query statement.
- Absence of file/network access and mutation.
- Always validate SQL before running it and inspect execution evidence afterward.
- Calculate numerical answers through executable queries.
- Do not ask the model to generate the final numerical answer from memory or prose.
- Do not silently change the model, truncate prompts, or choose an untested quantization.
- Never silently remove mandatory evidence to fit context.
- Treat empty results, zero, and `NULL` as distinct outcomes.
- Keep zero separate from blank, missing, and invalid values.
- Record ambiguous formats and failed conversions.
- Do not silently omit failed numeric conversions from aggregates.
- A percentage value of `25` means `25%`; multiplication by 100 occurs exactly once.
- The inference process cannot access evaluation answers.
- Public/private test answers must not drive training or tuning.
- Inference must run with Internet disabled, local model paths, and offline package installation.
- No automatic download fallback is permitted during inference.
- Preserve original data and record derived transformations.
- The model must preserve numerical meaning, negation, comparison operators, identifiers, and named entities.
- A failed search alone does not justify `unanswerable`.
- Stop when any relevant budget is exhausted.

## Where to look in the master plan

| Area | Section |
|---|---|
| Task, formats and conventions | §1 |
| Model and runtime profiles | §2 |
| Table detection, provenance, numeric and date parsing | §3.1–3.3 |
| Grain, joins and schema cards | §3.4–3.5 |
| Query normalization, retrieval and context overflow | §3.6–3.7 |
| Tools, parallel calls and dependencies | §4.1 and “Scheduling tool calls” |
| SQL generation, execution and validation | §4.2–4.3 |
| Budgets, throughput and answer decisions | §4.4–4.5 |
| Repo layout, Kaggle assets and checkpoints | §5 |
| Baselines, data splits, tests and gate acceptance | §6 |
