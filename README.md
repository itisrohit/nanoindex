# NanoIndex

A high-performance, lightweight vector similarity search engine built with **FastAPI**, **NumPy**, and **uv**.

---

## Key Features

- **Mathematical Clarity:** Python-only implementation of core search primitives (L2, Cosine).
- **Efficiency:** Utilizes memory-mapped files (`mmap`) for large-scale vector storage without excessive RAM usage.
- **Modern Stack:** Built with Python 3.10+, FastAPI, and Pydantic v2.
- **Developer First:** Fully typed with Mypy, formatted with Ruff, and managed with `uv`.

---

## Getting Started

### 1. Install System Requirements

NanoIndex requires `uv` for environment management and `just` as a task runner.

#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install just
- **macOS / Linux:** `brew install just` or `curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin`
- **Windows:** `choco install just` or `irm https://just.systems/install.ps1 | iex`

---

### 2. Installation

Clone the repository and synchronize the environment:

```bash
# Synchronize environment and install dependencies
just sync
```

---

## Development Workflow

We use `just` to automate common development tasks.

| Command | Description |
| :--- | :--- |
| `just dev` | Start the FastAPI development server (Port 8000) |
| `just check` | Run all quality checks (Ruff linter/formatter + Mypy) |
| `just fix` | Automatically fix linting issues and reformat code |
| `just test` | Execute the test suite using Pytest |
| `just update` | Upgrade dependencies and update the lockfile |
| `just clean` | Remove all temporary caches and virtual environments |

---

## API Documentation

Once the server is running (`just dev`), you can access the interactive documentation at:
- **Swagger UI:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **Redoc:** [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

---

## Project Structure

```text
nanoindex/
├── app/
│   ├── api/          # Route handlers and API versioning
│   ├── core/         # Configuration and settings
│   ├── internal/     # Mathematical primitives
│   ├── models/       # Pydantic schemas
│   └── services/     # Business logic and storage
├── data/             # Persistent vector storage (gitignored)
├── docs/             # Project documentation and roadmap
└── pyproject.toml    # Dependencies and tool configuration
```

---

## License

This project is licensed under the [MIT License](./LICENSE). See the file for full details.
