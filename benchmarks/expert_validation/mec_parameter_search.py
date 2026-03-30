from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.measures.metrics.proposed.mec import MEC
from src.measures.validation import minmax_normalize_x

if __package__ is None or __package__ == "":
    from benchmarks.expert_validation.data import ValidationData
else:
    from .data import ValidationData


@dataclass(frozen=True)
class SearchConfig:
    alpha_min: float = 0.5
    alpha_max: float = 3.0
    alpha_step: float = 0.01
    beta_min: float = 0.8
    beta_max: float = 2.0
    beta_step: float = 0.01
    top_k: int = 20
    output_dir: Path = Path(__file__).resolve().parent / "mec_parameter_search"


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be positive")
    if stop < start:
        raise ValueError("Grid stop must be greater than or equal to start")
    count = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(count), 10)


def load_validation_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = ValidationData()
    x_values = minmax_normalize_x(data.x_values.astype(np.float64))
    distributions = data.get_normalized_distributions().astype(np.float64)
    distributions = distributions / distributions.sum(axis=1, keepdims=True)
    expert_scores = data.expert_scores.astype(np.float64)
    return x_values, distributions, expert_scores


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(pearsonr(x, y).statistic)


def evaluate_parameters(
    alpha: float,
    beta: float,
    x_values: np.ndarray,
    distributions: np.ndarray,
    expert_scores: np.ndarray,
) -> dict[str, float]:
    measure = MEC(alpha=alpha, beta=beta)
    values = np.array([measure.compute(x_values, dist) for dist in distributions])

    kendall = float(kendalltau(expert_scores, values).statistic)
    spearman = float(spearmanr(expert_scores, values).statistic)
    pearson = safe_pearson(expert_scores, values)

    return {
        "alpha": alpha,
        "beta": beta,
        "kendall_tau": kendall,
        "spearman_rho": spearman,
        "pearson_r": pearson,
    }


def run_grid_search(config: SearchConfig) -> pd.DataFrame:
    x_values, distributions, expert_scores = load_validation_problem()
    alpha_values = inclusive_grid(config.alpha_min, config.alpha_max, config.alpha_step)
    beta_values = inclusive_grid(config.beta_min, config.beta_max, config.beta_step)

    rows: list[dict[str, float]] = []
    for alpha in alpha_values:
        for beta in beta_values:
            rows.append(
                evaluate_parameters(
                    alpha=float(alpha),
                    beta=float(beta),
                    x_values=x_values,
                    distributions=distributions,
                    expert_scores=expert_scores,
                )
            )

    results = pd.DataFrame(rows)
    return results.sort_values(
        by=["kendall_tau", "spearman_rho", "pearson_r", "alpha", "beta"],
        ascending=[False, False, False, True, True],
        ignore_index=True,
    )


def find_kendall_maxima(results: pd.DataFrame, atol: float = 1e-12) -> pd.DataFrame:
    best_tau = float(results["kendall_tau"].max())
    return results[np.isclose(results["kendall_tau"], best_tau, atol=atol)].copy()


def boundary_hits(
    config: SearchConfig, maxima: pd.DataFrame, atol: float = 1e-12
) -> list[str]:
    warnings: list[str] = []
    if np.isclose(maxima["alpha"], config.alpha_min, atol=atol).any():
        warnings.append(
            "The maximum touches alpha_min; consider expanding the alpha range downward."
        )
    if np.isclose(maxima["alpha"], config.alpha_max, atol=atol).any():
        warnings.append(
            "The maximum touches alpha_max; consider expanding the alpha range upward."
        )
    if np.isclose(maxima["beta"], config.beta_min, atol=atol).any():
        warnings.append(
            "The maximum touches beta_min; consider expanding the beta range downward."
        )
    if np.isclose(maxima["beta"], config.beta_max, atol=atol).any():
        warnings.append(
            "The maximum touches beta_max; consider expanding the beta range upward."
        )
    return warnings


def save_heatmap(results: pd.DataFrame, output_dir: Path) -> Path:
    pivot = results.pivot(index="beta", columns="alpha", values="kendall_tau")
    alpha_values = pivot.columns.to_numpy(dtype=float)
    beta_values = pivot.index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        extent=[
            alpha_values.min(),
            alpha_values.max(),
            beta_values.min(),
            beta_values.max(),
        ],
        cmap="viridis",
    )
    best = results.iloc[0]
    ax.scatter(best["alpha"], best["beta"], color="red", s=60, marker="x", linewidths=2)
    ax.set_title("Kendall tau against expert scores for MEC(alpha, beta)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    fig.colorbar(image, ax=ax, label="kendall_tau")
    fig.tight_layout()

    output_path = output_dir / "kendall_heatmap.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def format_table(rows: Iterable[dict[str, float]]) -> str:
    lines = [
        "| alpha | beta | kendall_tau | spearman_rho | pearson_r |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['alpha']:.4f} | "
            f"{row['beta']:.4f} | "
            f"{row['kendall_tau']:.12f} | "
            f"{row['spearman_rho']:.12f} | "
            f"{row['pearson_r']:.12f} |"
        )
    return "\n".join(lines)


def save_summary(
    config: SearchConfig,
    results: pd.DataFrame,
    maxima: pd.DataFrame,
    output_dir: Path,
    runtime_seconds: float,
    heatmap_path: Path,
) -> Path:
    warnings = boundary_hits(config, maxima)
    top_rows = results.head(config.top_k).to_dict(orient="records")
    maxima_alpha_min = float(maxima["alpha"].min())
    maxima_alpha_max = float(maxima["alpha"].max())
    maxima_beta_min = float(maxima["beta"].min())
    maxima_beta_max = float(maxima["beta"].max())

    summary = f"""# MEC Parameter Search

This report summarizes an exhaustive grid search for `MEC(alpha, beta)` against
expert judgments from Koudenburg et al. (2021), using Kendall tau as the
primary objective.

## Configuration

- alpha range: `{config.alpha_min}` to `{config.alpha_max}` with step `{config.alpha_step}`
- beta range: `{config.beta_min}` to `{config.beta_max}` with step `{config.beta_step}`
- grid points: `{len(results)}`
- runtime (seconds): `{runtime_seconds:.3f}`

## Best Kendall Tau

- max kendall_tau: `{results.iloc[0]["kendall_tau"]:.12f}`
- number of maxima on the grid: `{len(maxima)}`
- canonical best row after tie-break by Spearman, Pearson, alpha, beta:
  `alpha={results.iloc[0]["alpha"]:.4f}, beta={results.iloc[0]["beta"]:.4f}`
- maxima alpha range: `{maxima_alpha_min:.4f}` to `{maxima_alpha_max:.4f}`
- maxima beta range: `{maxima_beta_min:.4f}` to `{maxima_beta_max:.4f}`

### Top {config.top_k}

{format_table(top_rows)}

## Boundary Warnings

"""

    if warnings:
        summary += "\n".join(f"- {warning}" for warning in warnings)
    else:
        summary += "- No maximum lies on the search boundary."

    summary += f"""

## Artifacts

- Full grid: `grid_results.csv`
- Top-k: `top_results.csv`
- All Kendall maxima: `maxima.csv`
- Heatmap: `{heatmap_path.name}`

![Kendall heatmap]({heatmap_path.name})
"""

    output_path = output_dir / "summary.md"
    output_path.write_text(summary)
    return output_path


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustive grid search for MEC(alpha, beta) against expert judgments."
        )
    )
    parser.add_argument("--alpha-min", type=float, default=SearchConfig.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=SearchConfig.alpha_max)
    parser.add_argument("--alpha-step", type=float, default=SearchConfig.alpha_step)
    parser.add_argument("--beta-min", type=float, default=SearchConfig.beta_min)
    parser.add_argument("--beta-max", type=float, default=SearchConfig.beta_max)
    parser.add_argument("--beta-step", type=float, default=SearchConfig.beta_step)
    parser.add_argument("--top-k", type=int, default=SearchConfig.top_k)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SearchConfig.output_dir,
        help="Directory where CSV, plot, and summary artifacts will be written.",
    )
    args = parser.parse_args()
    return SearchConfig(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )


def main(config: SearchConfig | None = None) -> None:
    config = config or parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    results = run_grid_search(config)
    runtime_seconds = perf_counter() - start

    maxima = find_kendall_maxima(results)
    heatmap_path = save_heatmap(results, output_dir)

    results.to_csv(output_dir / "grid_results.csv", index=False)
    results.head(config.top_k).to_csv(output_dir / "top_results.csv", index=False)
    maxima.to_csv(output_dir / "maxima.csv", index=False)
    summary_path = save_summary(
        config, results, maxima, output_dir, runtime_seconds, heatmap_path
    )

    print(f"Saved full grid to {output_dir / 'grid_results.csv'}")
    print(f"Saved top results to {output_dir / 'top_results.csv'}")
    print(f"Saved maxima to {output_dir / 'maxima.csv'}")
    print(f"Saved heatmap to {heatmap_path}")
    print(f"Saved summary to {summary_path}")
    print("\nBest row:")
    print(results.iloc[0].to_dict())


if __name__ == "__main__":
    main()
