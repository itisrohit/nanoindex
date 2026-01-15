# justfile for NanoIndex
# Usage: just <recipe>

set shell := ["bash", "-c"]

# Default recipe: list all recipes
default:
    @just --list --unsorted

# --- Development ---

# Run the FastAPI development server
dev port="8000":
    uv run uvicorn app.main:app --reload --port {{port}}

# --- Quality ---

# Run all quality checks (linting, formatting, type checking)
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy .

# Fix all fixable lint issues and format the code
fix:
    uv run ruff check --fix .
    uv run ruff format .

# --- Dependencies ---

# Synchronize project dependencies and update the lockfile
sync:
    uv sync

# Update project dependencies
update:
    uv lock --upgrade
    uv sync

# --- Testing ---

# Run all unit and integration tests
test *args:
    uv run pytest {{args}}

# --- Cleanup ---

# Remove temporary files and caches
clean:
    rm -rf .venv .mypy_cache .ruff_cache .pytest_cache
    find . -type d -name "__pycache__" -exec rm -rf {} +
