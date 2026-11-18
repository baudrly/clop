#!/usr/bin/env python3
"""Inference throughput of every trained arm on CPU: 500 bp windows embedded per second (batch 256, best of three).

python scripts/speed.py results --threads 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from clop.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    sequences = ["".join(rng.choice(list("ACGT"), 500)) for _ in range(args.n)]
    out = {}
    for model in sorted(Path(args.results, "runs").glob("*/model.pt")):
        pipe = Pipeline.load(model)
        pipe.embed(sequences[:64])
        best = min(_time(pipe, sequences) for _ in range(3))
        out[model.parent.name] = {"windows_per_second": args.n / best, "parameters": pipe.model.n_parameters()}
        print(f"{model.parent.name:24s} {args.n / best:8.0f} windows/s")
    Path(args.results, "speed.json").write_text(json.dumps(out, indent=1))


def _time(pipe: Pipeline, sequences: list[str]) -> float:
    started = time.perf_counter()
    pipe.embed(sequences, batch_size=256)
    return time.perf_counter() - started


if __name__ == "__main__":
    main()
