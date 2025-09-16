# Repository Guidelines

## Project Structure & Module Organization
Core server code lives in `src/mcp_server_starrocks/`, with `server.py` exposing FastMCP resources and tools, `db_client.py` handling StarRocks connectivity, and `db_summary_manager.py` providing cached metadata helpers. Tests sit in `tests/`, mirroring the module layout (`test_db_client.py`). Top-level configs include `pyproject.toml` (package + dependency metadata), `pytest.ini` (test discovery flags), and `README.md` (server launch options). Assets such as `mcpserverdemo.jpg` and release notes live at the repo root.

## Build, Test, and Development Commands
- `uv pip install -e .[test]` — install the project in editable mode with test extras.
- `pytest` — run the default test suite (`-v --tb=short` is preconfigured via `pytest.ini`).
- `pytest --cov=src/mcp_server_starrocks` — optional coverage check when adding behaviors touching the StarRocks client.
- `uv run mcp-server-starrocks --test` — verify connection settings before starting an agent host.
- `uv run mcp-server-starrocks --mode streamable-http --port 8000` — start the development server for interactive testing.

## Coding Style & Naming Conventions
Target Python 3.10+. Follow PEP 8 with 4-space indentation, single quotes for short strings, and type hints for new public functions. Long literals and SQL strings should be wrapped or extracted to helpers. Reuse the `loguru` logger configured in `server.py` instead of `print`. Tool, resource, and function names remain descriptive but concise (e.g., `query_and_plotly_chart`).

## Testing Guidelines
Unit tests rely on `pytest`; add new files as `tests/test_<feature>.py` and functions as `test_<behavior>` to match `pytest.ini`. Mock StarRocks calls via fixtures or sample responses from `db_client.ResultSet`. Ensure regression tests cover both success and error paths, and update fixtures whenever wire formats change.

## Commit & Pull Request Guidelines
Commit messages follow an imperative, present-tense style (`add db summary`, `Enhance STARROCKS_URL parsing`). Group related changes in a single commit when possible. Pull requests should summarize intent, list environment variables touched (`STARROCKS_URL`, `STARROCKS_HOST`, etc.), reference tracking issues, and include screenshots or CLI transcripts if behavior or API responses change.

## Configuration Tips
Store credentials via environment variables (`STARROCKS_URL` or host/user/password trio). Use `STARROCKS_OVERVIEW_LIMIT` during testing to cap large schema dumps. Reset pooled connections via `reset_db_connections()` when swapping between environments to avoid stale sessions.
