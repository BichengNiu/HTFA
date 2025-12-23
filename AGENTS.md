# Repository Guidelines

## Project Structure & Module Organization
- `dashboard/` contains the Streamlit app and core Python modules.
  - `dashboard/app.py` is the main entrypoint.
  - `dashboard/core/` holds shared UI/backend helpers.
  - `dashboard/models/DFM/` covers data prep, training, evaluation, and decomposition.
  - `dashboard/analysis/` and `dashboard/preview/` host domain analysis and preview flows.
- `data/` stores input datasets (Excel/CSV/SQLite).
- `results/` holds exported models, plots, and aligned outputs.
- `docs/` contains papers and user documentation.
- Root tooling includes `requirements.txt`, `start.bat`, `Dockerfile`, and `docker-compose.yml`.

## Build, Test, and Development Commands
- `py -m pip install -r requirements.txt` installs runtime dependencies.
- `start.bat` runs the local Streamlit app, clears Python caches, and sets `HTFA_DEBUG_MODE=true`.
- `py -m streamlit run dashboard/app.py --server.port=8501` runs the app without the batch wrapper.
- `docker compose up --build` builds and runs the container on port `8501`.

## Coding Style & Naming Conventions
- Python, UTF-8, 4-space indentation; keep modules import-safe (avoid heavy side effects at import time).
- Use `snake_case` for functions/variables, `CamelCase` for classes, and `UPPER_SNAKE` for constants.
- Keep UI code under `dashboard/**/ui`, prep code under `dashboard/models/DFM/prep`, and training under `dashboard/models/DFM/train`.
- Avoid adding large binary artifacts unless they are expected outputs in `results/` or input fixtures in `data/`.

## Testing Guidelines
- No automated test suite is currently checked in, and no coverage target is documented.
- If you add tests, place them in `tests/` and name files `test_*.py`.
- Prefer `unittest` for zero extra dependencies; example: `py -m unittest discover -s tests`.

## Commit & Pull Request Guidelines
- Git history favors very short, action-oriented summaries (often in Chinese); keep messages brief and scoped.
- Example format: `fix: dfm training`, `optimize ui`, or `add prep step`.
- PRs should include a short summary, key files changed, and any manual verification steps.
- For Streamlit UI changes, include a screenshot or recording and note any data assumptions.

## Configuration & Data Notes
- `HTFA_DEBUG_MODE=true` enables a simplified local flow; set to `false` for production-like runs.
- Keep secrets out of `data/`; use environment variables for credentials or keys.
