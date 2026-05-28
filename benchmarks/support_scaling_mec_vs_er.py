#!/usr/bin/env python3
"""
Synthetic support-size scaling benchmark for MEC vs Esteban-Ray.

This benchmark is designed to test the asymptotic claim in the relevant
variable: the support size n of a single distribution. It differs from the
corpus benchmark, where there are millions of distributions but each one has
fixed support size n=5.

It reports average runtime per distribution for:
  - ER: current package EstebanRay implementation, O(n^2)
  - MEC(scipy): current package MEC implementation using scipy minimize_scalar
  - MEC(bisection): derivative-root bisection implementation for beta > 1,
    matching the O(n log(1/epsilon)) characterization

Example:
    python3 benchmarks/support_scaling_mec_vs_er.py
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measures.metrics.literature import EstebanRay  # noqa: E402
from measures.metrics.proposed import MEC  # noqa: E402
from measures.utils import mec_bisection_value  # noqa: E402


@dataclass(frozen=True)
class ScalingConfig:
    support_sizes: Tuple[int, ...]
    alpha_mec: float = 2.0
    beta_mec: float = 1.15
    alpha_er: float = 0.8
    epsilon: float = 1e-8
    seed: int = 20260528
    dtype: str = "float64"


@dataclass(frozen=True)
class TimingRow:
    measure: str
    support_size: int
    samples: int
    wall_seconds: float
    seconds_per_distribution: float
    distributions_per_second: float
    checksum: float
    loglog_slope: Optional[float] = None


@dataclass(frozen=True)
class ValidationRow:
    support_size: int
    checked_samples: int
    max_abs_diff_scipy_vs_bisection: float
    max_rel_diff_scipy_vs_bisection: float


def parse_support_sizes(value: str) -> Tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise ValueError("At least one support size is required")
    if any(size < 2 for size in sizes):
        raise ValueError("All support sizes must be at least 2")
    return sizes


def adaptive_samples(support_size: int) -> int:
    """Choose enough samples for stable timings while keeping large-n runs bounded."""
    if support_size <= 20:
        return 5000
    if support_size <= 50:
        return 2500
    if support_size <= 100:
        return 1500
    if support_size <= 200:
        return 800
    if support_size <= 400:
        return 350
    if support_size <= 800:
        return 150
    if support_size <= 1200:
        return 80
    if support_size <= 2000:
        return 40
    return 20


def resolve_sample_count(support_size: int, fixed_samples: Optional[int]) -> int:
    if fixed_samples is not None:
        if fixed_samples < 1:
            raise ValueError("--samples-per-size must be positive")
        return fixed_samples
    return adaptive_samples(support_size)


def generate_weights(
    rng: np.random.Generator, samples: int, support_size: int, dtype: np.dtype
) -> np.ndarray:
    """Generate positive normalized random masses."""
    weights = rng.exponential(scale=1.0, size=(samples, support_size)).astype(dtype)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def time_values(
    name: str,
    func: Callable[[np.ndarray], float],
    weights_matrix: np.ndarray,
) -> Tuple[float, float]:
    """Return elapsed seconds and checksum for one measure on all samples."""
    # Warm up once outside the timed region to avoid one-time overheads dominating
    # the smallest support sizes.
    if len(weights_matrix):
        _ = func(weights_matrix[0])

    checksum = 0.0
    start = perf_counter()
    for weights in weights_matrix:
        checksum += float(func(weights))
    elapsed = perf_counter() - start
    return elapsed, checksum


def fit_loglog_slopes(rows: List[TimingRow]) -> List[TimingRow]:
    """Fit one log-log slope per measure and attach it to every row for that measure."""
    rows_by_measure: Dict[str, List[TimingRow]] = {}
    for row in rows:
        rows_by_measure.setdefault(row.measure, []).append(row)

    slopes: Dict[str, float] = {}
    for measure, measure_rows in rows_by_measure.items():
        if len(measure_rows) < 2:
            slopes[measure] = float("nan")
            continue
        x = np.log(
            np.array([row.support_size for row in measure_rows], dtype=np.float64)
        )
        y = np.log(
            np.array(
                [row.seconds_per_distribution for row in measure_rows],
                dtype=np.float64,
            )
        )
        slopes[measure] = float(np.polyfit(x, y, deg=1)[0])

    return [
        TimingRow(
            measure=row.measure,
            support_size=row.support_size,
            samples=row.samples,
            wall_seconds=row.wall_seconds,
            seconds_per_distribution=row.seconds_per_distribution,
            distributions_per_second=row.distributions_per_second,
            checksum=row.checksum,
            loglog_slope=slopes[row.measure],
        )
        for row in rows
    ]


def validate_bisection_against_scipy(
    x: np.ndarray,
    weights_matrix: np.ndarray,
    mec_scipy: MEC,
    config: ScalingConfig,
    max_checks: int,
) -> ValidationRow:
    checked = min(max_checks, len(weights_matrix))
    abs_diffs = []
    rel_diffs = []
    for weights in weights_matrix[:checked]:
        scipy_value = float(mec_scipy.compute(x, weights))
        bisection_value = mec_bisection_value(
            x,
            weights,
            alpha=config.alpha_mec,
            beta=config.beta_mec,
            epsilon=config.epsilon,
        )
        abs_diff = abs(scipy_value - bisection_value)
        rel_diff = abs_diff / max(abs(scipy_value), 1e-15)
        abs_diffs.append(abs_diff)
        rel_diffs.append(rel_diff)

    return ValidationRow(
        support_size=len(x),
        checked_samples=checked,
        max_abs_diff_scipy_vs_bisection=float(max(abs_diffs) if abs_diffs else 0.0),
        max_rel_diff_scipy_vs_bisection=float(max(rel_diffs) if rel_diffs else 0.0),
    )


def run_benchmark(
    config: ScalingConfig,
    *,
    fixed_samples: Optional[int],
    validate_samples: int,
) -> Tuple[List[TimingRow], List[ValidationRow]]:
    rng = np.random.default_rng(config.seed)
    dtype = np.dtype(config.dtype)
    mec_scipy = MEC(alpha=config.alpha_mec, beta=config.beta_mec)
    er = EstebanRay(alpha=config.alpha_er)

    rows: List[TimingRow] = []
    validations: List[ValidationRow] = []

    for support_size in config.support_sizes:
        samples = resolve_sample_count(support_size, fixed_samples)
        x = np.linspace(0.0, 1.0, support_size, dtype=dtype)
        weights_matrix = generate_weights(rng, samples, support_size, dtype)

        print(f"support_size={support_size:,}, samples={samples:,}")

        measure_funcs: List[Tuple[str, Callable[[np.ndarray], float]]] = [
            (
                "mec_bisection",
                lambda w, x=x: mec_bisection_value(
                    x,
                    w,
                    alpha=config.alpha_mec,
                    beta=config.beta_mec,
                    epsilon=config.epsilon,
                ),
            ),
            ("mec_scipy", lambda w, x=x: mec_scipy.compute(x, w)),
            ("er", lambda w, x=x: er.compute(x, w)),
        ]

        for measure_name, func in measure_funcs:
            elapsed, checksum = time_values(measure_name, func, weights_matrix)
            seconds_per_distribution = elapsed / samples
            distributions_per_second = (
                samples / elapsed if elapsed > 0 else float("inf")
            )
            row = TimingRow(
                measure=measure_name,
                support_size=support_size,
                samples=samples,
                wall_seconds=elapsed,
                seconds_per_distribution=seconds_per_distribution,
                distributions_per_second=distributions_per_second,
                checksum=checksum,
            )
            rows.append(row)
            print(
                f"  {measure_name:>13}: "
                f"wall={elapsed:.6f}s, "
                f"per_dist={seconds_per_distribution * 1e6:.3f} us, "
                f"throughput={distributions_per_second:,.0f} dist/s"
            )

        validation = validate_bisection_against_scipy(
            x=x,
            weights_matrix=weights_matrix,
            mec_scipy=mec_scipy,
            config=config,
            max_checks=validate_samples,
        )
        validations.append(validation)
        print(
            "  validation: "
            f"max_abs_diff={validation.max_abs_diff_scipy_vs_bisection:.3e}, "
            f"max_rel_diff={validation.max_rel_diff_scipy_vs_bisection:.3e}"
        )
        print()

    return fit_loglog_slopes(rows), validations


def write_csv(rows: Sequence[TimingRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_validation_csv(rows: Sequence[ValidationRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_results(
    rows: Sequence[TimingRow], output_dir: Path, stem: str
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measures = ["mec_bisection", "mec_scipy", "er"]
    labels = {
        "mec_bisection": "MEC bisection",
        "mec_scipy": "MEC SciPy",
        "er": "ER",
    }
    colors = {
        "mec_bisection": "#1f77b4",
        "mec_scipy": "#ff7f0e",
        "er": "#2ca02c",
    }

    linear_path = output_dir / f"{stem}_linear.png"
    loglog_path = output_dir / f"{stem}_loglog.png"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for measure in measures:
        measure_rows = [row for row in rows if row.measure == measure]
        ax.plot(
            [row.support_size for row in measure_rows],
            [row.seconds_per_distribution * 1e6 for row in measure_rows],
            marker="o",
            label=labels[measure],
            color=colors[measure],
        )
    ax.set_title("Runtime per distribution as support size grows")
    ax.set_xlabel("Support size n")
    ax.set_ylabel("Microseconds per distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(linear_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for measure in measures:
        measure_rows = [row for row in rows if row.measure == measure]
        slope = measure_rows[0].loglog_slope if measure_rows else float("nan")
        ax.loglog(
            [row.support_size for row in measure_rows],
            [row.seconds_per_distribution * 1e6 for row in measure_rows],
            marker="o",
            label=f"{labels[measure]} (slope={slope:.2f})",
            color=colors[measure],
        )
    ax.set_title("Log-log runtime scaling with support size")
    ax.set_xlabel("Support size n")
    ax.set_ylabel("Microseconds per distribution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loglog_path, dpi=180)
    plt.close(fig)

    return linear_path, loglog_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic support-size scaling benchmark for MEC vs ER."
    )
    parser.add_argument(
        "--support-sizes",
        default="5,10,20,50,100,200,400,800,1200,1600,2000",
        help="Comma-separated support sizes to benchmark.",
    )
    parser.add_argument("--alpha-mec", type=float, default=2.0)
    parser.add_argument("--beta-mec", type=float, default=1.15)
    parser.add_argument("--alpha-er", type=float, default=0.8)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Bisection precision for MEC(bisection); default: 1e-8.",
    )
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--samples-per-size",
        type=int,
        default=None,
        help="Use a fixed sample count for every support size instead of adaptive counts.",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=10,
        help="Samples per support size used to validate bisection vs SciPy; default: 10.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float64", "float32"],
        default="float64",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "support_scaling_mec_vs_er",
    )
    parser.add_argument(
        "--stem",
        default="support_scaling",
        help="Output filename stem; default: support_scaling.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = ScalingConfig(
        support_sizes=parse_support_sizes(args.support_sizes),
        alpha_mec=args.alpha_mec,
        beta_mec=args.beta_mec,
        alpha_er=args.alpha_er,
        epsilon=args.epsilon,
        seed=args.seed,
        dtype=args.dtype,
    )

    print("Support-scaling benchmark configuration:")
    print(f"  support_sizes: {config.support_sizes}")
    print(f"  MEC alpha,beta: ({config.alpha_mec}, {config.beta_mec})")
    print(f"  ER alpha: {config.alpha_er}")
    print(f"  bisection epsilon: {config.epsilon}")
    print(f"  seed: {config.seed}")
    print(f"  dtype: {config.dtype}")
    print()

    rows, validations = run_benchmark(
        config,
        fixed_samples=args.samples_per_size,
        validate_samples=args.validate_samples,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.stem}.csv"
    validation_csv_path = output_dir / f"{args.stem}_validation.csv"
    json_path = output_dir / f"{args.stem}.json"

    write_csv(rows, csv_path)
    write_validation_csv(validations, validation_csv_path)
    linear_path, loglog_path = plot_results(rows, output_dir, args.stem)

    payload = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "repo_root": str(REPO_ROOT),
        },
        "config": asdict(config),
        "results": [asdict(row) for row in rows],
        "validation": [asdict(row) for row in validations],
        "artifacts": {
            "csv": str(csv_path),
            "validation_csv": str(validation_csv_path),
            "linear_plot": str(linear_path),
            "loglog_plot": str(loglog_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("Log-log fitted slopes:")
    for measure in sorted({row.measure for row in rows}):
        slope = next(row.loglog_slope for row in rows if row.measure == measure)
        print(f"  {measure}: {slope:.3f}")
    print()
    print(f"Wrote CSV:        {csv_path}")
    print(f"Wrote validation: {validation_csv_path}")
    print(f"Wrote JSON:       {json_path}")
    print(f"Wrote plot:       {linear_path}")
    print(f"Wrote plot:       {loglog_path}")


if __name__ == "__main__":
    main()
