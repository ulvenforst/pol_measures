#!/usr/bin/env python3
"""
Benchmark MEC vs Esteban-Ray over the generated opinion-distribution corpus.

The default corpus matches the comparison-matrix convention used elsewhere in
this repository: all k-bin compositions of `mass_size` with mirror distributions
omitted by keeping only the canonical representative `w <= reversed(w)`.

Examples:
    # Smoke test on a small prefix
    python3 benchmarks/corpus_timing_mec_vs_er.py --mass-size 100 --limit 10000 --no-output

    # Full canonical corpus, sequential core-compute timing
    python3 benchmarks/corpus_timing_mec_vs_er.py --mass-size 100 --workers 1

    # Auto-tune worker counts, then run the full corpus with the selected workers
    python3 benchmarks/corpus_timing_mec_vs_er.py --mass-size 100 --auto

    # Full canonical corpus using all logical CPU cores
    python3 benchmarks/corpus_timing_mec_vs_er.py --mass-size 100 --workers 0

    # Include public __call__ validation/normalization overhead
    python3 benchmarks/corpus_timing_mec_vs_er.py --mass-size 100 --workers 0 --use-call
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measures.metrics.literature import EstebanRay  # noqa: E402
from measures.metrics.proposed import MEC  # noqa: E402


@dataclass(frozen=True)
class CorpusConfig:
    mass_size: int = 100
    support_size: int = 5
    include_mirrors: bool = False
    limit: Optional[int] = None
    dtype: str = "float64"


@dataclass(frozen=True)
class MeasureConfig:
    mec_alpha: float = 2.0
    mec_beta: float = 1.15
    er_alpha: float = 0.8
    er_k: Optional[float] = None
    use_call: bool = False


@dataclass(frozen=True)
class TimingResult:
    measure: str
    distributions: int
    effective_evaluations: int
    support_size: int
    repeat: int
    workers: int
    chunk_size: int
    wall_seconds: float
    worker_compute_seconds: float
    seconds_per_distribution_wall: float
    distributions_per_second_wall: float
    checksum: float


def iter_count_vectors(total: int, bins: int) -> Iterator[Tuple[int, ...]]:
    """Yield all non-negative integer vectors of length `bins` summing to `total`."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if bins < 1:
        raise ValueError("bins must be positive")

    # Fast path for the default Likert-like corpus. Avoids recursion overhead for
    # the >4.5M compositions generated when total=100, bins=5.
    if bins == 5:
        for c0 in range(total + 1):
            remaining0 = total - c0
            for c1 in range(remaining0 + 1):
                remaining1 = remaining0 - c1
                for c2 in range(remaining1 + 1):
                    remaining2 = remaining1 - c2
                    for c3 in range(remaining2 + 1):
                        c4 = remaining2 - c3
                        yield (c0, c1, c2, c3, c4)
        return

    counts = [0] * bins

    def recurse(position: int, remaining: int) -> Iterator[Tuple[int, ...]]:
        if position == bins - 1:
            counts[position] = remaining
            yield tuple(counts)
            return

        for value in range(remaining + 1):
            counts[position] = value
            yield from recurse(position + 1, remaining - value)

    yield from recurse(0, total)


def count_compositions(total: int, bins: int) -> int:
    """Count all k-bin compositions of `total` via stars and bars."""
    return int(math.comb(total + bins - 1, bins - 1))


def count_palindromic_compositions(total: int, bins: int) -> int:
    """Count compositions fixed by reversal."""
    pair_count = bins // 2
    count = 0

    def recurse(pair_index: int, remaining: int) -> None:
        nonlocal count
        if pair_index == pair_count:
            if bins % 2 == 1:
                # The middle bin absorbs any remaining mass.
                count += 1
            elif remaining == 0:
                count += 1
            return

        # Each paired value consumes twice its mass.
        for value in range(remaining // 2 + 1):
            recurse(pair_index + 1, remaining - 2 * value)

    recurse(0, total)
    return count


def expected_corpus_size(config: CorpusConfig) -> int:
    total = count_compositions(config.mass_size, config.support_size)
    if config.include_mirrors:
        expected = total
    else:
        palindromic = count_palindromic_compositions(
            config.mass_size, config.support_size
        )
        expected = (total + palindromic) // 2

    if config.limit is not None:
        return min(config.limit, expected)
    return expected


def is_canonical(counts: Sequence[int]) -> bool:
    return tuple(counts) <= tuple(reversed(counts))


def build_corpus(config: CorpusConfig) -> Tuple[np.ndarray, float, int]:
    """
    Materialize the corpus as a dense array of normalized weights.

    For the default n=100,k=5 canonical corpus this is about 2.3M x 5 float64,
    roughly 92 MB, which is intentionally small for a high-memory workstation.
    """
    if config.mass_size <= 0:
        raise ValueError("mass_size must be positive")
    if config.support_size < 2:
        raise ValueError("support_size must be at least 2")

    dtype = np.dtype(config.dtype)
    expected = expected_corpus_size(config)
    corpus = np.empty((expected, config.support_size), dtype=dtype)

    start = perf_counter()
    row = 0
    visited = 0
    for counts in iter_count_vectors(config.mass_size, config.support_size):
        visited += 1
        if not config.include_mirrors and not is_canonical(counts):
            continue

        corpus[row] = counts
        row += 1
        if config.limit is not None and row >= config.limit:
            break

    corpus = corpus[:row]
    corpus /= config.mass_size
    elapsed = perf_counter() - start
    return corpus, elapsed, visited


def make_measure(measure_name: str, config: MeasureConfig):
    if measure_name == "mec":
        return MEC(alpha=config.mec_alpha, beta=config.mec_beta)
    if measure_name == "er":
        return EstebanRay(alpha=config.er_alpha, K=config.er_k)
    raise ValueError(f"Unknown measure: {measure_name}")


def compute_chunk(
    measure_name: str,
    x: np.ndarray,
    weights_chunk: np.ndarray,
    repeat: int,
    measure_config: MeasureConfig,
) -> Dict[str, float]:
    """Compute one chunk and return timing metadata."""
    measure = make_measure(measure_name, measure_config)
    checksum = 0.0

    start = perf_counter()
    for _ in range(repeat):
        for weights in weights_chunk:
            if measure_config.use_call:
                value = measure(x, weights)
                if not isinstance(value, (int, float, np.floating)):
                    raise TypeError(
                        "Expected scalar result from measure(x, weights) when labels=None"
                    )
            else:
                value = measure.compute(x, weights)
            checksum += float(value)
    elapsed = perf_counter() - start

    return {
        "evaluations": float(len(weights_chunk) * repeat),
        "elapsed": elapsed,
        "checksum": checksum,
    }


def iter_chunk_bounds(total_rows: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, total_rows, chunk_size):
        yield start, min(start + chunk_size, total_rows)


def benchmark_measure(
    measure_name: str,
    corpus: np.ndarray,
    support_size: int,
    repeat: int,
    workers: int,
    chunk_size: int,
    measure_config: MeasureConfig,
) -> TimingResult:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    x = np.linspace(0.0, 1.0, support_size, dtype=corpus.dtype)
    total_rows = len(corpus)
    worker_compute_seconds = 0.0
    checksum = 0.0
    evaluations = total_rows * repeat

    wall_start = perf_counter()
    if workers == 1:
        for start, end in iter_chunk_bounds(total_rows, chunk_size):
            partial = compute_chunk(
                measure_name=measure_name,
                x=x,
                weights_chunk=corpus[start:end],
                repeat=repeat,
                measure_config=measure_config,
            )
            worker_compute_seconds += partial["elapsed"]
            checksum += partial["checksum"]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    compute_chunk,
                    measure_name,
                    x,
                    corpus[start:end],
                    repeat,
                    measure_config,
                )
                for start, end in iter_chunk_bounds(total_rows, chunk_size)
            ]
            for future in as_completed(futures):
                partial = future.result()
                worker_compute_seconds += partial["elapsed"]
                checksum += partial["checksum"]

    wall_seconds = perf_counter() - wall_start
    seconds_per_distribution = (
        wall_seconds / evaluations if evaluations else float("nan")
    )
    throughput = evaluations / wall_seconds if wall_seconds > 0 else float("inf")

    return TimingResult(
        measure=measure_name,
        distributions=total_rows,
        effective_evaluations=evaluations,
        support_size=support_size,
        repeat=repeat,
        workers=workers,
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        worker_compute_seconds=worker_compute_seconds,
        seconds_per_distribution_wall=seconds_per_distribution,
        distributions_per_second_wall=throughput,
        checksum=checksum,
    )


def resolve_workers(requested_workers: int) -> int:
    if requested_workers == 0:
        return max(os.cpu_count() or 1, 1)
    if requested_workers < 0:
        raise ValueError("workers must be >= 0; use 0 for all logical CPUs")
    return max(requested_workers, 1)


def parse_measure_list(value: str) -> List[str]:
    if value == "both":
        return ["mec", "er"]
    return [value]


def parse_worker_candidates(value: str) -> List[int]:
    """Parse comma-separated worker candidates; 0 means all logical CPUs."""
    candidates: List[int] = []
    seen = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        requested = int(part)
        resolved = resolve_workers(requested)
        if resolved in seen:
            continue
        candidates.append(resolved)
        seen.add(resolved)

    if not candidates:
        raise ValueError("At least one worker candidate is required")
    return candidates


def tune_workers(
    measure_name: str,
    corpus: np.ndarray,
    support_size: int,
    candidate_workers: Sequence[int],
    tune_limit: int,
    tune_repeat: int,
    tune_chunk_size: int,
    measure_config: MeasureConfig,
) -> Tuple[int, List[TimingResult]]:
    """Benchmark worker candidates on a prefix and choose best wall throughput."""
    if tune_limit < 1:
        raise ValueError("tune_limit must be positive")
    if tune_repeat < 1:
        raise ValueError("tune_repeat must be at least 1")
    if tune_chunk_size < 1:
        raise ValueError("tune_chunk_size must be at least 1")

    tune_rows = min(tune_limit, len(corpus))
    tune_corpus = corpus[:tune_rows]
    results: List[TimingResult] = []

    print(f"Tuning {measure_name.upper()} on {tune_rows:,} distributions...")
    for workers in candidate_workers:
        result = benchmark_measure(
            measure_name=measure_name,
            corpus=tune_corpus,
            support_size=support_size,
            repeat=tune_repeat,
            workers=workers,
            chunk_size=tune_chunk_size,
            measure_config=measure_config,
        )
        results.append(result)
        print(f"  workers={workers:>2}: ", end="")
        print_result(result)

    best = min(results, key=lambda result: result.wall_seconds)
    print(f"Selected workers for {measure_name.upper()}: {best.workers}\n")
    return best.workers, results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark total time for MEC and Esteban-Ray over the generated "
            "distribution corpus."
        )
    )
    parser.add_argument(
        "--mass-size",
        "--n",
        type=int,
        default=100,
        help="Total integer mass distributed across bins; default: 100.",
    )
    parser.add_argument(
        "--support-size",
        "--k",
        type=int,
        default=5,
        help="Number of support points/bins; default: 5.",
    )
    parser.add_argument(
        "--include-mirrors",
        action="store_true",
        help="Use all compositions instead of the canonical mirror-pruned corpus.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only benchmark the first LIMIT generated distributions.",
    )
    parser.add_argument(
        "--measure",
        choices=["both", "mec", "er"],
        default="both",
        help="Which measure(s) to run; default: both.",
    )
    parser.add_argument("--mec-alpha", type=float, default=2.0)
    parser.add_argument("--mec-beta", type=float, default=1.15)
    parser.add_argument("--er-alpha", type=float, default=0.8)
    parser.add_argument(
        "--er-k",
        type=float,
        default=None,
        help="Optional explicit Esteban-Ray normalization constant K.",
    )
    parser.add_argument(
        "--use-call",
        action="store_true",
        help="Benchmark public measure(x, w) calls instead of compute(x, w).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each measure over the corpus this many times; default: 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process workers. Use 0 for all logical CPUs; default: 1.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Tune worker counts on a corpus prefix, then run each selected measure "
            "on the full requested corpus with its best worker count."
        ),
    )
    parser.add_argument(
        "--worker-candidates",
        default="1,4,8,12,14,16,0",
        help=(
            "Comma-separated worker candidates used by --auto. "
            "Use 0 for all logical CPUs; default: 1,4,8,12,14,16,0."
        ),
    )
    parser.add_argument(
        "--tune-limit",
        type=int,
        default=300_000,
        help="Corpus-prefix size for --auto tuning; default: 300000.",
    )
    parser.add_argument(
        "--tune-repeat",
        type=int,
        default=1,
        help="Repeat count for --auto tuning evaluations; default: 1.",
    )
    parser.add_argument(
        "--tune-chunk-size",
        type=int,
        default=5_000,
        help="Distributions per worker task during --auto tuning; default: 5000.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25_000,
        help="Distributions per worker task for final runs; default: 25000.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float64", "float32"],
        default="float64",
        help="Corpus array dtype; default: float64.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "corpus_timing_mec_vs_er" / "results.json",
        help="JSON output path. A CSV summary with the same stem is also written.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write JSON/CSV artifacts.",
    )
    return parser


def write_outputs(payload: Dict[str, object], output_path: Path) -> Tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    csv_path = output_path.with_suffix(".csv")
    rows = payload["results"]
    assert isinstance(rows, list)
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")

    return output_path, csv_path


def print_result(result: TimingResult) -> None:
    print(
        f"{result.measure.upper():>3}: "
        f"wall={result.wall_seconds:.6f}s, "
        f"worker_sum={result.worker_compute_seconds:.6f}s, "
        f"throughput={result.distributions_per_second_wall:,.0f} dist/s, "
        f"per_dist={result.seconds_per_distribution_wall * 1e6:.3f} us"
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    corpus_config = CorpusConfig(
        mass_size=args.mass_size,
        support_size=args.support_size,
        include_mirrors=args.include_mirrors,
        limit=args.limit,
        dtype=args.dtype,
    )
    measure_config = MeasureConfig(
        mec_alpha=args.mec_alpha,
        mec_beta=args.mec_beta,
        er_alpha=args.er_alpha,
        er_k=args.er_k,
        use_call=args.use_call,
    )
    workers = resolve_workers(args.workers)
    measures = parse_measure_list(args.measure)
    worker_candidates = parse_worker_candidates(args.worker_candidates)

    full_count = count_compositions(corpus_config.mass_size, corpus_config.support_size)
    expected_count = expected_corpus_size(corpus_config)
    print("Corpus configuration:")
    print(f"  mass_size: {corpus_config.mass_size}")
    print(f"  support_size: {corpus_config.support_size}")
    print(f"  include_mirrors: {corpus_config.include_mirrors}")
    print(f"  full compositions: {full_count:,}")
    print(f"  benchmark distributions: {expected_count:,}")
    print(f"  dtype: {corpus_config.dtype}")
    print()

    corpus, generation_seconds, visited_compositions = build_corpus(corpus_config)
    memory_mb = corpus.nbytes / (1024**2)
    print(
        f"Built corpus: {len(corpus):,} distributions, "
        f"visited={visited_compositions:,}, "
        f"memory={memory_mb:.2f} MiB, "
        f"generation={generation_seconds:.6f}s"
    )
    print(
        f"Benchmark mode: {'public __call__' if args.use_call else 'core compute'}, "
        f"default_workers={workers}, chunk_size={args.chunk_size:,}, repeat={args.repeat}"
    )
    if args.auto:
        print(
            f"Auto tuning: candidates={worker_candidates}, "
            f"tune_limit={args.tune_limit:,}, "
            f"tune_chunk_size={args.tune_chunk_size:,}, "
            f"tune_repeat={args.tune_repeat}"
        )
    print()

    results: List[TimingResult] = []
    tuning_results: List[dict] = []
    for measure_name in measures:
        selected_workers = workers
        if args.auto:
            selected_workers, tuning = tune_workers(
                measure_name=measure_name,
                corpus=corpus,
                support_size=corpus_config.support_size,
                candidate_workers=worker_candidates,
                tune_limit=args.tune_limit,
                tune_repeat=args.tune_repeat,
                tune_chunk_size=args.tune_chunk_size,
                measure_config=measure_config,
            )
            tuning_results.append(
                {
                    "measure": measure_name,
                    "selected_workers": selected_workers,
                    "results": [asdict(result) for result in tuning],
                }
            )

        print(f"Running full {measure_name.upper()} with workers={selected_workers}...")
        result = benchmark_measure(
            measure_name=measure_name,
            corpus=corpus,
            support_size=corpus_config.support_size,
            repeat=args.repeat,
            workers=selected_workers,
            chunk_size=args.chunk_size,
            measure_config=measure_config,
        )
        results.append(result)
        print_result(result)

    result_by_name = {result.measure: result for result in results}
    if "mec" in result_by_name and "er" in result_by_name:
        mec = result_by_name["mec"]
        er = result_by_name["er"]
        ratio = (
            er.wall_seconds / mec.wall_seconds if mec.wall_seconds > 0 else float("inf")
        )
        faster = "MEC" if mec.wall_seconds < er.wall_seconds else "ER"
        print()
        print(
            f"Wall-time ratio ER/MEC: {ratio:.4f} "
            f"({faster} faster for this corpus/configuration)"
        )

    payload: Dict[str, object] = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "repo_root": str(REPO_ROOT),
        },
        "corpus": {
            **asdict(corpus_config),
            "full_compositions": full_count,
            "benchmark_distributions": len(corpus),
            "visited_compositions": visited_compositions,
            "generation_seconds": generation_seconds,
            "memory_mib": memory_mb,
        },
        "measure_config": asdict(measure_config),
        "benchmark_config": {
            "default_workers": workers,
            "chunk_size": args.chunk_size,
            "repeat": args.repeat,
            "mode": "call" if args.use_call else "compute",
            "auto": bool(args.auto),
            "worker_candidates": worker_candidates,
            "tune_limit": args.tune_limit,
            "tune_repeat": args.tune_repeat,
            "tune_chunk_size": args.tune_chunk_size,
        },
        "tuning": tuning_results,
        "results": [asdict(result) for result in results],
    }

    if not args.no_output:
        json_path, csv_path = write_outputs(payload, args.output)
        print()
        print(f"Wrote JSON: {json_path}")
        print(f"Wrote CSV:  {csv_path}")


if __name__ == "__main__":
    main()
