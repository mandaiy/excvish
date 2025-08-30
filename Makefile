.PHONY: format
format:
	@uv run ruff format
	@uv run ruff check --fix

.PHONY: lint
lint:
	@uv run ruff format --check
	@uv run ruff check
	@uv run ty check src
