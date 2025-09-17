"""
Concurrency benchmark with --chaos option: spawn tasks and randomly cancel some.
"""

import asyncio
import random
import argparse
import json
from pathlib import Path
import time


async def worker(i, delay=0.01):
    await asyncio.sleep(delay)
    return i


async def run(n_tasks=1000, chaos_rate=0.0):
    tasks = [asyncio.create_task(worker(i, delay=0.001)) for i in range(n_tasks)]
    if chaos_rate > 0:
        # randomly cancel some tasks
        to_cancel = random.sample(tasks, int(len(tasks) * chaos_rate))
        for t in to_cancel:
            t.cancel()
    start = time.time()
    done = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.time() - start
    return {"n_tasks": n_tasks, "chaos_rate": chaos_rate, "duration": duration, "done": len(done)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--chaos", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="bench_concurrency.json")
    args = parser.parse_args()
    res = asyncio.run(run(args.n, args.chaos))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
