"""
bench_compare.py

Simple benchmark harness to compare Python vs Rust (if fed_crypto is built).
Writes bench_results.json with timings.
"""

import time
import json
import numpy as np
from pathlib import Path
import argparse

RESULTS = {"runs": []}


def bench_python_shamir(secret_len=32, n=10, t=5, reps=100):
    from FedGNN_advanced.privacy.shamir import split_secret_bytes, reconstruct_secret_bytes
    s = b"\x01" * secret_len
    t0 = time.time()
    for _ in range(reps):
        shares = split_secret_bytes(s, n, t)
        reconstruct_secret_bytes(shares[:t], secret_len)
    t1 = time.time()
    return t1 - t0


def main(out="bench_results.json"):
    res = {}
    res["python_shamir_time"] = bench_python_shamir()
    # If fed_crypto exists, attempt to import and bench
    try:
        import fed_crypto  # type: ignore
        t0 = time.time()
        for _ in range(100):
            fed_crypto.bench_shamir()
        t1 = time.time()
        res["rust_shamir_time"] = t1 - t0
    except Exception:
        res["rust_shamir_time"] = None
    Path(out).write_text(json.dumps(res, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
