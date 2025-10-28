# Repository Guidelines


## Project Structure & Module Organization

`dashboard/` contains the Streamlit application, with `app.py` as the entry point and feature modules under `analysis/`, `core/`, `explore/`, and `ui/`. Modeling pipelines live in `dashboard/models/DFM/`, where `train/` houses the training coordinator (`training/trainer.py`) and associated utilities. Keep source-controlled datasets in `data/` read-only and treat `docs/` as the reference library for stakeholder deliverables.

## Build, Test, and Development Commands

- `python -m venv .venv && .venv\\Scripts\\activate`: create and activate the local virtual environment.
- `pip install -r requirements.txt`: install the Streamlit and modeling dependencies listed for the dashboard.
- `py -m streamlit run dashboard/app.py --server.port=8501`: launch the dashboard directly (mirrors `start.bat` cleanup and port checks).
- `pytest dashboard/models/DFM/train/tests`: run the current regression suite for the dynamic factor model training flow.

## Coding Style & Naming Conventions

Follow the existing Python conventions: 4-space indentation, lowercase `_` module and function names, and PascalCase for classes. Keep imports absolute (see `dashboard/models/DFM/train/training/trainer.py`) and favor type hints plus concise docstrings for public APIs. Align log messages with the structured patterns provided by `dashboard/models/DFM/train/utils/logger.py`, and avoid introducing notebooks or ad-hoc scripts inside `dashboard/`.

## Testing Guidelines

Unit and integration tests reside beside their targets (e.g., `dashboard/models/DFM/train/tests`). Name test files `test_<feature>.py` and isolate fixtures inside the same package to keep discovery fast. Run `pytest --maxfail=1 --disable-warnings` before pushing, and capture coverage deltas whenever modeling logic changes; add targeted fixtures for new model states rather than broad mocks.

## Commit & Pull Request Guidelines

Keep commit titles within 72 characters, preferring the `<scope>: <change>` pattern seen in history (`OpenSpec: …`, `文档: …`, `完成模型训练重构`). Group related edits into a single commit and include concise bilingual summaries when touching user-facing text. Pull requests should link the relevant tracking ticket, outline validation (commands run, datasets touched), and attach updated screenshots or metrics whenever UI or forecast outputs change.

## Data & Security Notes

Treat files under `data/` as production-quality inputs; do not commit regenerated datasets without clearance. Sanitize credentials by relying on environment variables when extending `dashboard/auth/`. When sharing previews, strip personally identifiable customer data and use the `preview/` module’s synthetic fixtures instead.
