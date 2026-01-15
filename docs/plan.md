# NanoIndex: Lightweight Vector Similarity Search Engine
> *A minimal, clear, and Pythonic reimplementation of core vector similarity search primitives.*

---

## Table of Contents
1. [Overview & Objectives](#1-overview--objectives)
2. [Scope & Constraints](#2-scope--constraints)
3. [Architecture Design](#3-architecture-design)
4. [Project Roadmap](#4-project-roadmap)
5. [Implementation Standards](#5-implementation-standards)
6. [Strategic Positioning](#6-strategic-positioning)
7. [Success Metrics](#7-success-metrics)

---

## 1. Overview & Objectives

### The Vision
The primary goal of **NanoIndex** is to build a high-fidelity, readable, Python-only reimplementation of the core algorithms behind industry-standard vector similarity search. This project prioritizes **mathematical clarity** and **algorithmic correctness** over raw scale or GPU acceleration.

### Core Motivation
- **Foundational Understanding:** Deconstruct the "black box" of vector databases.
- **First Principles:** Implement the essential primitives of modern AI infrastructure.
- **Adaptive Systems:** Demonstrate how search engines self-optimize using agentic loops.
- **Educational Value:** Provide a codebase that serves as a living documentation of vector search logic.

---

## 2. Scope & Constraints

To maintain project focus and ensure a transparent implementation, the following architectural elements are **explicitly out of scope**:

| Category | Excluded Features |
| :--- | :--- |
| **Hardware** | GPU / CUDA Support, SIMD Hardware Optimizations |
| **Scaling** | Billion-scale indexing, Distributed Search |
| **Complexity** | HNSW, Product Quantization (PQ), Optimized PQ (OPQ) |
| **Integration** | External-facing products, Managed services |

**Note:** A thin local API layer (FastAPI) is included for testing, benchmarking, and inspection—it serves as a harness, not the product.

---

## 3. Architecture Design

### Data Flow
NanoIndex follows a streamlined pipeline for vector retrieval:
`Vectors` ➔ `Index Storage` ➔ `Distance Computation` ➔ `Top-K Selection`

### Directory Structure
```text
nanoindex/
├── app/
│   ├── main.py             # FastAPI entry point
│   ├── api/                # API Endpoints (v1)
│   ├── core/               # Configuration and Settings
│   ├── internal/           # Core math (Distance, etc.)
│   ├── models/             # Pydantic schemas
│   └── services/           # DataStore and Indexing logic
├── data/                   # Persistent vector storage
├── docs/                   # Documentation
├── tests/                  # Unit and Integration tests
└── pyproject.toml          # Dependency management (uv/hatch)
```
```

---

## 4. Project Roadmap

### Phase 1: Mathematical Foundation (Completed)
*Implementation of core distance metrics.*
- [x] **Focus:** `app/internal/distance.py`
- [x] **Tasks:** L2 distance, Cosine similarity, Vector normalization.
- [x] **Success Criteria:** Functions validated against NumPy primitives; Passing Mypy strict checks.

### Phase 2: Indexing Architecture (Completed)
*Development of the vector storage layer.*
- [x] **Focus:** `app/services/datastore.py`
- [x] **Tasks:** Memory-mapped (`mmap`) NumPy storage, Integer ID mapping.
- [x] **Success Criteria:** Conceptually equivalent to a flat index structure with persistence.

### Phase 3: Retrieval Logic (Completed)
*Execution of the search algorithm.*
- [x] **Focus:** `app/services/searcher.py` and `app/api/v1/endpoints/search.py`
- [x] **Tasks:** Comprehensive query-to-index distance calculation, Top-K selection logic.
- [x] **Success Criteria:** Deterministic nearest neighbor retrieval for both L2 and Cosine metrics via API.

### Phase 4: Performance Analysis (Completed)
*Benchmarking and validation.*
- [x] **Focus:** `scripts/benchmark.py`
- [x] **Tasks:** Batch query processing, Latency measurement, Scalability testing (1k to 100k vectors).
- [x] **Success Criteria:** Documented p95 latency and ingestion throughput in `docs/benchmarks.md`.

### Phase 5: Technical Extension (Completed: Simple IVF)
*Implementation of an Inverted File Index.*
- [x] **Focus:** `app/internal/clustering.py` and `app/services/indexer.py`
- [x] **Tasks:** Mini K-Means implementation, Centroid-based partitioning, `nprobe` search logic.
- [x] **Success Criteria:** Functional IVF search demonstrating partitioned retrieval with measurable speedup.

### Phase 6: Performance Optimization (Completed: Practical Enhancements)
*Practical optimizations inspired by established vector databases.*
- [x] **Focus:** `app/services/datastore.py`, `app/internal/distance.py`, `app/services/indexer.py`
- [x] **Tasks:** 
  - Squared norm caching in DataStore
  - Batch quantization for cell assignment
  - K-Means subsampling for large datasets
  - Search budget (`max_codes`) implementation
- [x] **Success Criteria:** Achieved **1.51x speedup** on 150k vectors with IVF search.

### Phase 7: Self-Tuning Search Agent (Completed)
*Adaptive query optimization using Multi-Armed Bandit (MAB) algorithm.*
- [x] **Focus:** `app/services/adaptive.py`, `app/services/searcher.py`
- [x] **Core Concept:** Implement an **epsilon-greedy** or **UCB1** bandit agent that learns optimal search strategies
- [x] **Tasks:**
  - Design search strategy "arms": (Flat, IVF-conservative, IVF-aggressive, IVF-balanced)
  - Track per-arm metrics: latency, recall quality proxy, resource usage
  - Implement exploration-exploitation balance (epsilon decay or UCB confidence bounds)
  - Add latency constraint awareness (SLA-based arm selection)
  - Persist agent state (arm statistics, rewards history)
  - Integrate into SearchService with minimal overhead
- [x] **Success Criteria:** Agent automatically selects optimal strategy based on query patterns, achieving better average latency than static configuration while maintaining quality.

**USP Statement:**  
*NanoIndex includes a self-tuning search agent that dynamically adapts search strategy (flat vs IVF, nprobe, scan budget) based on observed query behavior and latency constraints.*

**Technical Approach:**
- **Multi-Armed Bandit (MAB):** Each search configuration is an "arm"
- **Reward Signal:** Inverse latency (faster = higher reward) with quality penalty
- **Exploration:** Epsilon-greedy (10% random) or UCB1 (confidence-based)
- **Adaptation:** Update arm statistics after each query
- **Persistence:** Save arm performance to survive restarts

**Implementation Highlights:**
- 11 comprehensive unit tests (17 total tests passing)
- Epsilon-greedy and UCB1 algorithms implemented
- State persistence with automatic loading
- `/agent/stats` and `/agent/reset` endpoints
- Full documentation in `docs/adaptive-agent.md`

---

## 5. Implementation Standards

1. **Clarity Over Optimization:** Code must be readable by a junior engineer; avoid "clever" one-liners.
2. **Atomic Functions:** Each function should encapsulate a single mathematical or logical concept.
3. **Intentional Documentation:** Comments should explain *why* a specific approach was taken, rather than *what* the code is doing.
4. **Simplicity First:** If a logic block appears overly complex, it must be refactored for simplicity.

---

## 6. Strategic Positioning

### How to Present NanoIndex
The project should be positioned as a **fundamental research/learning tool**. It is designed to demonstrate technical depth and understanding of retrieval systems.

### Core Value Proposition
### Core Value Proposition
By building from scratch and studying established implementations, NanoIndex provides insight into the trade-offs between accuracy, complexity, and performance—and demonstrates how **agentic loops** can automate these trade-offs dynamically.

---

## 7. Success Metrics

The project is considered "Done" when the following conditions are met:
- [x] **Infrastructural:** Foundation (uv, ruff, mypy, just) established and clean.
- [x] **Functional:** Flat index and Top-K search are fully operational.
- [x] **Advanced:** IVF index with practical optimizations achieving 1.5x+ speedup.
- [x] **Adaptive:** Self-tuning search agent using Multi-Armed Bandit for automatic strategy selection.
- [x] **Verified:** Comprehensive unit and integration test suite (17 passing tests).
- [x] **Lean:** Codebase remains focused and maintainable.
- [x] **Transparent:** The documentation accurately reflects the project's educational nature and optimization journey.

- [x] **Transparent:** The documentation accurately reflects the project's educational nature and optimization journey.

