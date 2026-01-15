# System Limits & Failure Modes

NanoIndex is designed to be **transparent**. Part of that transparency is acknowledging where the system breaks, where it performs poorly, and why certain architectural decisions (like avoiding HNSW) were made.

This document serves as a "Negative Capabilities" report—defining what the system *cannot* or *should not* do.

## 1. When IVF Hurts Performance

While Inverted File Index (IVF) is generally faster for large datasets, it is **not always the optimal strategy**.

### The Overhead Threshold
For small datasets (typically < 10,000 vectors), the computational cost of:
1.  Computing distances to centroids (coarse quantization)
2.  Loading and scanning candidate lists (fine quantization)

...often exceeds the cost of a simple brute-force (Flat) scan.

**Failure Mode:** Enabling IVF on a dataset of 500 vectors.
**Result:** Higher latency than flat search due to overhead, with potentially lower recall.
**Mitigation:** The self-tuning agent detects this latency penalty and naturally favors the "Flat" arm for small datasets.

## 2. Agent Misalignment

The **Self-Tuning Agent** optimizes for **latency**, using it as a proxy for "success." This creates specific blind spots.

### The "Fast but Wrong" Trap
If an aggressive IVF strategy (e.g., `nprobe=1`, `max_codes=100`) returns results in 0.1ms but misses the true nearest neighbor, the agent currently sees this as a *high reward* outcome (Reward = 1000/0.1 = 10,000).

**Failure Mode:** The agent converges on an ultra-fast strategy that provides poor recall.
**Result:** User gets fast, irrelevant results.
**Constraint:** NanoIndex assumes "Balanced" and "Conservative" arms have acceptable baseline recall. The agent purely optimizes speed within those "safe" bounds.

## 3. Memory Mapping (mmap) Pitfalls

NanoIndex relies heavily on `numpy.memmap` to handle datasets larger than RAM. This delegate's memory management to the OS page cache.

### Cold Start Latency
When the system first starts, or after a long period of inactivity, the index files are not in the OS page cache.

**Failure Mode:** The first few queries trigger massive page faults as data is read from disk.
**Result:** P99 latency spikes (e.g., 500ms vs 20ms).
**Mitigation:** "Warm-up" queries are necessary in production environments, though NanoIndex (being a research tool) does not automate this.

### The NUMA Trap
On multi-socket systems (modern servers), mmap performance can degrade if memory is allocated on a remote NUMA node. NumPy does not inherently manage NUMA affinity.

## 4. Why No HNSW?

Hierarchical Navigable Small World (HNSW) graphs are the industry standard for vector search (used in Milvus, Pinecone, Weaviate). NanoIndex explicitly avoids HNSW.

**Reasoning:**
1.  **Complexity:** Implementing a correct, thread-safe HNSW graph requires ~2,000+ LOC of complex pointer manipulation and graph traversal logic. This violates the project's goal of *readability*.
2.  **Memory Overhead:** HNSW requires maintaining a massive graph structure in memory (neighbors per node), often consuming 2-3x more RAM than the raw vectors.
3.  **Educational Value:** IVF is easier to reason about (Partitioning -> Pruning), whereas HNSW behaves more like a randomized routing algorithm.

## 5. Concurrency Limits

NanoIndex uses a single global lock for write operations (`DataStore` resizing).

**Failure Mode:** Heavy concurrent write loads (e.g., 100 threads adding vectors).
**Result:** The Global Interpreter Lock (GIL) and the write lock serialize all ingestion, leading to throughput collapse.
**Design Trade-off:** NanoIndex optimizes for *read* performance and simple *bulk* ingestion, not high-concurrency real-time updates.
