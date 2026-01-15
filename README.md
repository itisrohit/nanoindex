<div align="center">

# NanoIndex

A single-node vector similarity search engine that implements flat and IVF-style indexing for exact and partitioned nearest-neighbor retrieval, designed for clarity and learning rather than scale.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

</div>

---

## Project Status

> **Status:** NanoIndex is **feature-complete** as a learning-focused vector similarity engine. Future work is intentionally limited to documentation and minor refactoring to preserve clarity and scope.

---

## Key Features

- **Advanced Indexing:** IVF (Inverted File Index) with K-Means partitioning for sub-linear search performance
- **Self-Tuning Agent:** Adaptive search agent using Multi-Armed Bandit algorithms (epsilon-greedy, UCB1) to automatically optimize search strategy
- **Performance Optimized:** Achieves 1.51x speedup on 150k vectors through squared norm caching, batch quantization, and search budget control
- **Mathematical Clarity:** Python-only implementation of core search primitives (L2, Cosine)
- **Efficient Storage:** Memory-mapped files (`mmap`) for large-scale vector storage without excessive RAM usage
- **Modern Stack:** Built with Python 3.10+, FastAPI, and Pydantic v2
- **Developer First:** Fully typed with Mypy, formatted with Ruff, and managed with `uv`

---

## Getting Started

### 1. Install System Requirements

#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install just

**macOS / Linux:**
```bash
brew install just
```
or
```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

**Windows:**
```bash
choco install just
```
or
```powershell
irm https://just.systems/install.ps1 | iex
```

---

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/itisrohit/nanoindex.git
cd nanoindex

# Synchronize environment and install dependencies
just sync
```

---

### 3. Run the Server

```bash
# Start the development server
just dev
```

The API will be available at `http://localhost:8000`

---

## Performance Benchmarks

### Search Performance (150k vectors, 128D)

| Metric | Flat Search | IVF Search | Speedup |
|--------|-------------|------------|---------|
| Average Latency | 30.51ms | 20.23ms | **1.51x** |
| Query Volume | 50 queries | 50 queries | - |
| Top-K | 10 | 10 | - |

**Key Optimizations:**
- **Self-Tuning Agent:** Automatically selects best search strategy (Flat vs IVF) based on latency constraints
- Squared norm caching for faster distance computation
- Batch quantization using matrix multiplication
- K-Means subsampling for efficient training
- Search budget control for predictable latency

See [`docs/benchmarks.md`](./docs/benchmarks.md) for detailed performance analysis.

---

## Development Workflow

| Command | Description |
| :--- | :--- |
| `just dev` | Start the FastAPI development server (Port 8000) |
| `just check` | Run all quality checks (Ruff linter/formatter + Mypy) |
| `just fix` | Automatically fix linting issues and reformat code |
| `just test` | Execute the test suite using Pytest |
| `just reset` | Reset the vector index (clear all data) |
| `just update` | Upgrade dependencies and update the lockfile |
| `just clean` | Remove all temporary caches and virtual environments |

---

## API Documentation

Once the server is running (`just dev`), you can access the interactive documentation at:
- **Swagger UI:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **Redoc:** [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

### Core Endpoints

#### Add Vectors
```bash
POST /api/v1/index/add
{
  "vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "ids": [1, 2]
}
```

#### Train IVF Index
```bash
POST /api/v1/index/train?n_cells=200
```

#### Search
```bash
POST /api/v1/search
{
  "vector": [0.1, 0.2, ...],
  "top_k": 10,
  "use_index": true
}
```

---

## Running Benchmarks

```bash
# Start the server in one terminal
just dev

# Run benchmarks in another terminal
uv run python scripts/benchmark.py
```

---

## Documentation

- **[Project Plan](./docs/plan.md)** - Architecture, roadmap, and design decisions
- **[Benchmarks](./docs/benchmarks.md)** - Detailed performance analysis
- **[Adaptive Agent](./docs/adaptive-agent.md)** - Self-tuning agent design and algorithms
- **[System Limits](./docs/failure_modes.md)** - Failure modes and engineering trade-offs
- **[API Docs](http://localhost:8000/api/v1/docs)** - Interactive API documentation (when server is running)

---

## Project Goals

NanoIndex is designed as a **fundamental research and learning tool** to demonstrate:
- How vector similarity search works under the hood
- Trade-offs between accuracy, complexity, and performance
- Advanced optimization techniques for vector search
- Clean, maintainable Python architecture
- **System Limits:** [Failure Modes & Trade-offs](./docs/failure_modes.md) documented explicitly

---

## License

This project is licensed under the [MIT License](./LICENSE).
