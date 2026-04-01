.PHONY: dev lint format

UV := uv run

# Install/sync dev dependencies (run once or after dependency changes).
dev:
	uv sync --group dev

# Format with ruff (replaces black + isort). Writes files in place.
format:
	$(UV) ruff format .
	$(UV) ruff check . --fix --select I

# Run ruff (format + lint) then mypy. Writes formatting fixes in place.
# A passing `make lint` guarantees CI will accept the code.
# To fully mirror CI (yaml checks, uv-lock, etc.) run: uv run pre-commit run --all-files
lint: format
	$(UV) ruff check .
	$(UV) mypy -p aieng
