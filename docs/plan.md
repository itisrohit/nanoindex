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
- **Educational Value:** Provide a codebase that serves as a living documentation of vector search logic.

---

## 2. Scope & Constraints

To maintain project focus and ensure a transparent implementation, the following architectural elements are **explicitly out of scope**:

| Category | Excluded Features |
| :--- | :--- |
| **Hardware** | GPU / CUDA Support, SIMD Hardware Optimizations |
| **Scaling** | Billion-scale indexing, Distributed Search |
| **Complexity** | HNSW, Product Quantization (PQ), Optimized PQ (OPQ) |
| **Integration** | Training Models, Web APIs, Graphical User Interfaces |

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
*Target codebase size: 300–500 lines of highly documented Python.*

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

### Phase 4: Performance Analysis
*Benchmarking and validation.*
- **Focus:** `benchmark.py`
- **Tasks:** Batch query processing, Latency measurement, Scalability testing (1k to 100k vectors).
- **Metrics:** Latency, Throughput, Memory Footprint.

### Phase 5: Technical Extension (Select One)
- **Option A (Efficiency):** Simple IVF (Inverted File Index) with centroid partitioning.
- **Option B (Transparency):** Explainable Search (Decomposing distance contributions).
- **Option C (Resource Tracking):** Granular memory analysis per vector.

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
By building from scratch, NanoIndex provides insight into the trade-offs between accuracy, complexity, and performance that are often hidden by production-grade libraries.

---

## 7. Success Metrics

The project is considered "Done" when the following conditions are met:
- [x] **Infrastructural:** Foundation (uv, ruff, mypy, just) established and clean.
- [ ] **Functional:** Flat index and Top-K search are fully operational.
- [ ] **Verified:** Benchmark results are reproducible and documented.
- [ ] **Lean:** Total Lines of Code (LOC) remain under 500.
- [ ] **Transparent:** The documentation accurately reflects the project's educational nature.

---

**Final Mindset:**
This project is an exercise in **deep understanding**. Reimplementing fundamentals is the hallmark of professional engineering growth—whether in databases, compilers, or AI systems.

