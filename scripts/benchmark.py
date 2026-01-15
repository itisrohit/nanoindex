"""
Performance benchmark for NanoIndex including IVF comparison.
"""

import time

import httpx
import numpy as np

BASE_URL = "http://localhost:8000/api/v1"


def benchmark_ingestion(
    num_vectors: int, dimension: int, batch_size: int = 1000
) -> float:
    print(f"Benchmarking Ingestion: {num_vectors} vectors, dim={dimension}...")
    start_time = time.perf_counter()

    for i in range(0, num_vectors, batch_size):
        vectors = np.random.rand(batch_size, dimension).astype("float32").tolist()
        ids = list(range(i, i + batch_size))
        response = httpx.post(
            f"{BASE_URL}/index/add/",
            json={"vectors": vectors, "ids": ids},
            timeout=30.0,
        )
        if response.status_code != 200:
            print(f"Error adding vectors: {response.status_code} - {response.text}")
            return 0.0

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(
        f"Ingested {num_vectors} vectors in {duration:.2f}s ({num_vectors / duration:.2f} vectors/s)"
    )
    return duration


def benchmark_search(
    num_queries: int, dimension: int, top_k: int = 10, use_index: bool = True
) -> float:
    mode = "IVF" if use_index else "Flat"
    print(f"Benchmarking Search ({mode}): {num_queries} queries, top_k={top_k}...")
    latencies: list[float] = []

    for _ in range(num_queries):
        query = np.random.rand(dimension).astype("float32").tolist()
        start_time = time.perf_counter()
        response = httpx.post(
            f"{BASE_URL}/search/",
            json={"vector": query, "top_k": top_k, "use_index": use_index},
            timeout=10.0,
        )
        end_time = time.perf_counter()
        if response.status_code == 200:
            latencies.append((end_time - start_time) * 1000)
        else:
            print(f"Search error: {response.status_code} - {response.text}")

    if not latencies:
        print("No successful queries performed.")
        return 0.0

    avg_latency = np.mean(latencies)
    print(f"{mode} Average Latency: {avg_latency:.2f}ms")
    return float(avg_latency)


def train_index(n_cells: int = 100) -> None:
    print(f"Training IVF Index with {n_cells} cells...")
    start_time = time.perf_counter()
    response = httpx.post(f"{BASE_URL}/index/train?n_cells={n_cells}", timeout=60.0)
    duration = time.perf_counter() - start_time
    if response.status_code == 200:
        print(f"Index trained in {duration:.2f}s")
    else:
        print(f"Training failed: {response.text}")


if __name__ == "__main__":
    # Settings (increased size for more visible speedup)
    TOTAL_VECTORS = 50000
    DIMENSION = 128
    NUM_QUERIES = 50

    try:
        # Reset first
        httpx.delete(f"{BASE_URL}/index/reset")

        benchmark_ingestion(TOTAL_VECTORS, DIMENSION)

        # 1. Flat Search Baseline
        flat_avg = benchmark_search(NUM_QUERIES, DIMENSION, use_index=False)

        # 2. Train and IVF Search
        train_index(n_cells=200)
        ivf_avg = benchmark_search(NUM_QUERIES, DIMENSION, use_index=True)

        if flat_avg > 0 and ivf_avg > 0:
            speedup = flat_avg / ivf_avg
            print("\nSummary:")
            print(f"Flat Latency: {flat_avg:.2f}ms")
            print(f"IVF Latency:  {ivf_avg:.2f}ms")
            print(f"Speedup:      {speedup:.2f}x")

    except Exception as e:
        print(
            f"Benchmark failed: {e}. Make sure the server is running with 'just dev'."
        )
