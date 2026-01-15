"""
Performance benchmark for NanoIndex.
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


def benchmark_search(num_queries: int, dimension: int, top_k: int = 10) -> float:
    print(f"Benchmarking Search: {num_queries} queries, top_k={top_k}...")
    latencies: list[float] = []

    for _ in range(num_queries):
        query = np.random.rand(dimension).astype("float32").tolist()
        start_time = time.perf_counter()
        response = httpx.post(
            f"{BASE_URL}/search/", json={"vector": query, "top_k": top_k}, timeout=10.0
        )
        end_time = time.perf_counter()
        if response.status_code == 200:
            # We use the wall-clock time including network overhead for the benchmark
            latencies.append((end_time - start_time) * 1000)
        else:
            print(f"Search error: {response.status_code} - {response.text}")

    if not latencies:
        print("No successful queries performed.")
        return 0.0

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"p95 Latency: {p95_latency:.2f}ms")
    return float(avg_latency)


if __name__ == "__main__":
    # Settings
    TOTAL_VECTORS = 10000
    DIMENSION = 128
    NUM_QUERIES = 100

    try:
        # Check health first
        httpx.get(f"{BASE_URL}/health")

        benchmark_ingestion(TOTAL_VECTORS, DIMENSION)
        benchmark_search(NUM_QUERIES, DIMENSION)
    except Exception as e:
        print(
            f"Benchmark failed: {e}. Make sure the server is running with 'just dev'."
        )
