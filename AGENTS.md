# Repository Guidelines

## Project Structure & Module Organization
- Source: `config.py`, `dataset.py`, `model.py`, `train.py`.
- Scripts: `analyze_data_length.py`, `truncation_stats.py`, `test_data_loading.py`.
- Assets: `tokenizer_en.json`, `tokenizer_zh.json`.
- Keep new utilities in the root alongside related modules; prefer small, focused files.

## Build, Test, and Development Commands
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -r requirements.txt`.
- Run training: `python train.py`.
- Dataset check: `python test_data_loading.py` (prints dataset stats).
- Data audits: `python analyze_data_length.py`, `python truncation_stats.py`.

## Coding Style & Naming Conventions
- Python 3.9+; use 4‑space indentation, PEP 8 defaults.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Type hints where useful; docstrings for public functions/classes.
- Keep functions small; isolate I/O (downloads, file writes) from core logic for testability.
- Optional formatters: use Black (line length 88) and isort if installed.

## Testing Guidelines
- Framework: lightweight tests under `test_*.py` in repo root.
- Conventions: test files `test_*.py`, test functions `test_*`.
- Run: `python test_data_loading.py` or, if using pytest, `pytest -q`.
- Prefer assertions over prints for new tests; include edge cases (empty inputs, long sequences, truncation behavior).

## Commit & Pull Request Guidelines
- Commit message: imperative mood, concise subject, optional body explaining rationale.
  - Example: `feat(data): add length histogram to audit script`.
- Scope changes narrowly; separate refactors from behavior changes.
- PRs should include: clear description, linked issues, reproduction or before/after notes, and sample commands.

## Security & Configuration Tips
- Configuration lives in `config.py`; avoid hard‑coding secrets. Use env vars (e.g., `HF_ENDPOINT`) when needed.
- Do not commit large trained weights or datasets; keep tokenizer JSONs but exclude new bulky artifacts.
- Networked downloads rely on Hugging Face `datasets`; prefer mirrors via env if required.

## Architecture Overview
- Data: loaded via Hugging Face `datasets` in `dataset.py`.
- Model: sequence‑to‑sequence Transformer in `model.py`.
- Training: loop and orchestration in `train.py`; parameters from `config.py`.
