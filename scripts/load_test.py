#!/usr/bin/env python3
"""
Lightweight load/smoke test for the deployed inference server.

Fires concurrent requests at /predict with random (but correctly
shaped) input windows and reports latency percentiles and error rate.
This is intentionally dependency-light (just `requests` + stdlib
`concurrent.futures`) rather than pulling in a full load-testing
framework like Locust, since the goal here is a quick sanity check
you can run right after deploying, not a full performance benchmark.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --n-requests 200 --concurrency 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke/load test the ForgeNet inference server")
    p.add_argument("--url", type=str, default="http://localhost:8000")
    p.add_argument("--n-requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--num-features", type=int, default=6)
    return p.parse_args()


def make_request(url: str, seq_len: int, num_features: int) -> tuple[bool, float]:
    payload = {"series": np.random.randn(seq_len, num_features).tolist()}
    start = time.perf_counter()
    try:
        resp = requests.post(f"{url}/predict", json=payload, timeout=10)
        ok = resp.status_code == 200
    except requests.RequestException:
        ok = False
    elapsed = (time.perf_counter() - start) * 1000
    return ok, elapsed


def main() -> None:
    args = parse_args()

    health = requests.get(f"{args.url}/health", timeout=5)
    print(f"health check: {health.status_code} {health.json()}")

    latencies = []
    errors = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(make_request, args.url, args.seq_len, args.num_features)
            for _ in range(args.n_requests)
        ]
        for fut in as_completed(futures):
            ok, elapsed = fut.result()
            latencies.append(elapsed)
            if not ok:
                errors += 1

    latencies.sort()
    n = len(latencies)
    print(f"\nrequests: {n} | errors: {errors} ({errors / n:.1%})")
    print(f"latency (ms) -- p50: {latencies[int(n*0.5)]:.1f}  "
          f"p90: {latencies[int(n*0.9)]:.1f}  "
          f"p99: {latencies[min(int(n*0.99), n-1)]:.1f}  "
          f"mean: {statistics.mean(latencies):.1f}")


if __name__ == "__main__":
    main()
