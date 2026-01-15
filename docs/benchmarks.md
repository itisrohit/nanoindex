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

Search benchmarks analyze the time taken to retrieve the Top-K nearest neighbors using both Flat (exhaustive) and IVF (partitioned) search strategies.

### Baseline Results (50k vectors)

| Metric | Measurement |
| :--- | :--- |
| Query Volume | 50 Queries |
| Top-K | 10 |
| Flat Average Latency | ~18.54 ms |
| IVF Average Latency | ~17.10 ms |
| Speedup | **1.08x** |

### Optimized Results (150k vectors)

| Metric | Measurement |
| :--- | :--- |
| Query Volume | 50 Queries |
| Top-K | 10 |
| IVF Cells | 200 |
| Flat Average Latency | ~30.51 ms |
| IVF Average Latency | ~20.23 ms |
| Speedup | **1.51x** |

### Performance Optimizations

The following techniques were implemented to improve search efficiency:

1. **Squared Norm Caching**  
   Pre-compute and store `||v||²` for all vectors to eliminate redundant calculations during distance computation.

2. **Batch Quantization**  
   Use matrix multiplication (`dist² = ||v||² + ||c||² - 2·v·c`) for cell assignment instead of iterative loops.

3. **K-Means Subsampling**  
   Train clustering on a representative subset of vectors (max 10k samples), then assign all data points to the final centroids.

4. **Search Budget Control**  
   Limit the maximum number of vectors scanned per query (`max_codes = 50k`) to ensure predictable query latency.

**Key Insight:** IVF speedup becomes more pronounced as dataset size increases. At 150k vectors, we observe a **1.51x improvement** over exhaustive search, with the gap widening further at larger scales.

---

## 4. Ablation Study: Agent Impact

To validate the **Self-Tuning Search Agent**, we compared static configurations against adaptive agent strategies on the 150k vector dataset.

| Configuration | Strategy Description | Average Latency | Impact |
| :--- | :--- | :--- | :--- |
| **Flat Search** | Exhaustive scan (Baseline) | ~30.51 ms | Baseline |
| **IVF Static** | Fixed "Balanced" config (nprobe=10) | ~20.23 ms | **1.51x Faster** |
| **Agent (ε=0)** | Pure Exploitation (Converged to "Conservative") | ~15.50 ms | **1.96x Faster** |
| **Agent (ε=0.1)** | Adaptive (90% Best / 10% Explore) | ~17.10 ms | **1.78x Faster** |

**Conclusion:** The agent successfully identified that the "IVF Conservative" arm (`nprobe=5`) offered the best latency for this dataset, outperforming the static "Balanced" default. Even with exploration enabled (ε=0.1), the agent maintained a significant speed advantage while retaining the ability to adapt to changing conditions.

---

## 5. Resource Usage

NanoIndex is designed for a low memory footprint relative to dataset size.

- **Disk Usage:** Approximately 100MB per 200,000 vectors (at 128 dimensions).
- **RAM Usage:** Minimal, as the operating system handles the paging of memory-mapped files.

---

## 6. Instructions for Replication

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
