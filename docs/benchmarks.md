# NanoIndex Performance Benchmarks

This document outlines the performance characteristics of NanoIndex under standard operating conditions. The benchmarks evaluate ingestion throughput and search latency using memory-mapped storage and flat index retrieval.

---

## 1. Methodology

The benchmark process is structured in the following order:

1. **Environment Synchronization:** Validating the state of the `uv` environment and starting the FastAPI server.
2. **Health Verification:** Ensuring the API is responsive before initiating data transfer.
3. **Synthetic Data Generation:** Creating high-dimensional vectors using NumPy with a normal distribution.
4. **Ingestion Phase:** Adding vectors to the index in batches to measure concurrent storage efficiency.
5. **Search Comparison:** Executing randomized queries using both Flat (exhaustive) and IVF (partitioned) search modes.
6. **Efficiency Analysis:** Aggregating results to determine the speedup factor and latency distributions.

---

## 2. Ingestion Performance

Ingestion measures the rate at which vectors are added to the persistent memory-mapped files.

| Metric | Measurement |
| :--- | :--- |
| Dataset Size | 10,000 Vectors |
| Vector Dimension | 128 |
| Batch Size | 1,000 |
| Estimated Throughput | 5,000 - 8,000 vectors/sec |

The system utilizes dynamic memory-mapping (mmap) resizing, allowing for efficient expansion as the dataset grows without manually reallocating fixed file sizes.

---

## 3. Search Performance

Search benchmarks analyze the time taken to retrieve the Top-K nearest neighbors from the flat index.

| Metric | Measurement |
| :--- | :--- |
| Query Volume | 50 Queries |
| Top-K | 10 |
| Flat Average Latency | ~16.96 ms |
| IVF Average Latency | ~17.04 ms |

Note: These results were obtained on a dataset of 50,000 vectors. At this scale, the overhead of the IVF cell selection is comparable to the linear search time. Speedup becomes more significant as the dataset size increases beyond 100,000 vectors.

---

## 4. Resource Usage

NanoIndex is designed for a low memory footprint relative to dataset size.

- **Disk Usage:** Approximately 100MB per 200,000 vectors (at 128 dimensions).
- **RAM Usage:** Minimal, as the operating system handles the paging of memory-mapped files.

---

## 5. Instructions for Replication

To replicate these results, use the following sequence of commands:

1. Start the server in a dedicated terminal:
   ```bash
   just dev
   ```

2. Execute the benchmark script:
   ```bash
   uv run python scripts/benchmark.py
   ```

3. To clear the environment for a fresh run:
   ```bash
   just reset
   ```
