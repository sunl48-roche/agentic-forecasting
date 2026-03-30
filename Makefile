.PHONY: dev lint format

UV := uv run

# Install/sync dev dependencies (run once or after dependency changes).
dev:
	uv sync --group dev

# Code quality: Black, isort, and mypy. Run `make dev lint` after a fresh clone.
lint:
	$(UV) black --check .
	$(UV) isort --check-only .
	$(UV) mypy -p aieng

# Apply Black and isort formatting.
format:
	$(UV) black .
	$(UV) isort .
