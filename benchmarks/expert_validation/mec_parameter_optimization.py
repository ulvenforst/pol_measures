from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

if __package__ is None or __package__ == "":
    from benchmarks.expert_validation.mec_parameter_search import (
        SearchConfig,
        find_kendall_maxima,
        load_validation_problem,
        run_grid_search,
        save_heatmap,
    )
else:
    from .mec_parameter_search import (
        SearchConfig,
        find_kendall_maxima,
        load_validation_problem,
        run_grid_search,
        save_heatmap,
    )

from src.measures.metrics.proposed.mec import MEC


@dataclass(frozen=True)
class OptimizationConfig:
    alpha_min: float = 0.5
    alpha_max: float = 3.0
    alpha_step: float = 0.01
    beta_min: float = 0.8
    beta_max: float = 2.0
    beta_step: float = 0.01
    refinement_seed_top_k: int = 25
    refinement_alpha_step: float = 0.002
    refinement_beta_step: float = 0.002
    stability_candidate_top_k: int = 100
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42
    top_k: int = 20
    output_dir: Path = Path(__file__).resolve().parent / "mec_parameter_optimization"


def safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    statistic = kendalltau(x, y).statistic
    return float("nan") if statistic is None else float(statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    statistic = spearmanr(x, y).statistic
    return float("nan") if statistic is None else float(statistic)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    statistic = pearsonr(x, y).statistic
    return float("nan") if statistic is None else float(statistic)


def build_search_config(
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    output_dir: Path,
) -> SearchConfig:
    return SearchConfig(
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        output_dir=output_dir,
    )


def build_refined_search_config(
    coarse_results: pd.DataFrame,
    config: OptimizationConfig,
    coarse_config: SearchConfig,
    output_dir: Path,
) -> SearchConfig:
    seed_rows = coarse_results.head(config.refinement_seed_top_k)

    alpha_min = max(
        coarse_config.alpha_min,
        float(seed_rows["alpha"].min()) - coarse_config.alpha_step,
    )
    alpha_max = min(
        coarse_config.alpha_max,
        float(seed_rows["alpha"].max()) + coarse_config.alpha_step,
    )
    beta_min = max(
        coarse_config.beta_min, float(seed_rows["beta"].min()) - coarse_config.beta_step
    )
    beta_max = min(
        coarse_config.beta_max, float(seed_rows["beta"].max()) + coarse_config.beta_step
    )

    return build_search_config(
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=config.refinement_alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=config.refinement_beta_step,
        output_dir=output_dir,
    )


def compute_values_grid(
    candidates: pd.DataFrame, x_values: np.ndarray, distributions: np.ndarray
) -> np.ndarray:
    values = np.empty((len(candidates), len(distributions)), dtype=np.float64)
    for row_index, row in enumerate(candidates.itertuples(index=False)):
        measure = MEC(alpha=float(row.alpha), beta=float(row.beta))
        values[row_index] = np.array(
            [measure.compute(x_values, dist) for dist in distributions],
            dtype=np.float64,
        )
    return values


def candidate_statistics(
    candidates: pd.DataFrame,
    candidate_values: np.ndarray,
    expert_scores: np.ndarray,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    n_candidates, n_distributions = candidate_values.shape
    bootstrap_indices = np.random.default_rng(bootstrap_seed).integers(
        0, n_distributions, size=(bootstrap_resamples, n_distributions)
    )

    loo_masks = [np.arange(n_distributions) != idx for idx in range(n_distributions)]

    bootstrap_tau_matrix = np.empty(
        (n_candidates, bootstrap_resamples), dtype=np.float64
    )
    loo_tau_matrix = np.empty((n_candidates, n_distributions), dtype=np.float64)

    rows: list[dict[str, float]] = []
    for idx, row in enumerate(candidates.itertuples(index=False)):
        values = candidate_values[idx]

        full_kendall = safe_kendall(expert_scores, values)
        full_spearman = safe_spearman(expert_scores, values)
        full_pearson = safe_pearson(expert_scores, values)

        for bootstrap_idx, sample_idx in enumerate(bootstrap_indices):
            bootstrap_tau_matrix[idx, bootstrap_idx] = safe_kendall(
                expert_scores[sample_idx], values[sample_idx]
            )

        for loo_idx, mask in enumerate(loo_masks):
            loo_tau_matrix[idx, loo_idx] = safe_kendall(
                expert_scores[mask], values[mask]
            )

        rows.append(
            {
                "alpha": float(row.alpha),
                "beta": float(row.beta),
                "full_kendall_tau": full_kendall,
                "full_spearman_rho": full_spearman,
                "full_pearson_r": full_pearson,
                "bootstrap_mean_kendall": float(np.nanmean(bootstrap_tau_matrix[idx])),
                "bootstrap_std_kendall": float(np.nanstd(bootstrap_tau_matrix[idx])),
                "bootstrap_min_kendall": float(np.nanmin(bootstrap_tau_matrix[idx])),
                "loo_mean_kendall": float(np.nanmean(loo_tau_matrix[idx])),
                "loo_std_kendall": float(np.nanstd(loo_tau_matrix[idx])),
                "loo_min_kendall": float(np.nanmin(loo_tau_matrix[idx])),
            }
        )

    stats = pd.DataFrame(rows)
    win_share = np.zeros(n_candidates, dtype=np.float64)
    for bootstrap_idx in range(bootstrap_resamples):
        column = bootstrap_tau_matrix[:, bootstrap_idx]
        finite = np.isfinite(column)
        if not finite.any():
            continue
        best_tau = np.nanmax(column)
        winners = np.flatnonzero(np.isclose(column, best_tau, atol=1e-12))
        win_share[winners] += 1.0 / len(winners)

    stats["bootstrap_win_rate"] = win_share / bootstrap_resamples
    return stats.sort_values(
        by=[
            "full_kendall_tau",
            "bootstrap_mean_kendall",
            "bootstrap_win_rate",
            "loo_mean_kendall",
            "full_spearman_rho",
            "full_pearson_r",
            "alpha",
            "beta",
        ],
        ascending=[False, False, False, False, False, False, True, True],
        ignore_index=True,
    )


def format_table(results: pd.DataFrame, columns: list[str]) -> str:
    headers = {
        "alpha": "alpha",
        "beta": "beta",
        "full_kendall_tau": "full_kendall_tau",
        "bootstrap_mean_kendall": "bootstrap_mean_kendall",
        "bootstrap_win_rate": "bootstrap_win_rate",
        "loo_mean_kendall": "loo_mean_kendall",
        "full_spearman_rho": "full_spearman_rho",
        "full_pearson_r": "full_pearson_r",
    }
    header_row = "| " + " | ".join(headers[col] for col in columns) + " |"
    rule_row = "| " + " | ".join("---:" for _ in columns) + " |"
    lines = [header_row, rule_row]
    for row in results[columns].itertuples(index=False):
        formatted: list[str] = []
        for value in row:
            if isinstance(value, (int, float, np.floating)):
                formatted.append(f"{float(value):.12f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def save_summary(
    config: OptimizationConfig,
    coarse_config: SearchConfig,
    refined_config: SearchConfig,
    coarse_results: pd.DataFrame,
    refined_results: pd.DataFrame,
    stability_candidates: pd.DataFrame,
    stability_results: pd.DataFrame,
    coarse_runtime: float,
    refined_runtime: float,
    stability_runtime: float,
    output_dir: Path,
) -> Path:
    coarse_maxima = find_kendall_maxima(coarse_results)
    refined_maxima = find_kendall_maxima(refined_results)
    recommended = stability_results.iloc[0]
    bootstrap_tied = bool(
        np.isclose(
            stability_results["bootstrap_mean_kendall"],
            stability_results["bootstrap_mean_kendall"].iloc[0],
            atol=1e-12,
        ).all()
    )
    loo_tied = bool(
        np.isclose(
            stability_results["loo_mean_kendall"],
            stability_results["loo_mean_kendall"].iloc[0],
            atol=1e-12,
        ).all()
    )
    selection_note = (
        "Bootstrap and leave-one-out do not separate the shortlisted candidates; "
        "the final recommendation is therefore the candidate with the best "
        "full-sample Pearson correlation inside the shortlisted frontier."
        if bootstrap_tied and loo_tied
        else "Bootstrap and leave-one-out contribute to separating the shortlisted candidates."
    )

    summary = f"""# MEC Parameter Optimization

This experiment optimizes `MEC(alpha, beta)` against expert judgments using a
three-stage pipeline:

1. exhaustive coarse search on the full parameter range
2. exhaustive refined search around the best coarse region
3. stability analysis on the refined Kendall maxima via bootstrap and leave-one-out

## Coarse Search

- alpha range: `{coarse_config.alpha_min}` to `{coarse_config.alpha_max}` step `{
        coarse_config.alpha_step
    }`
- beta range: `{coarse_config.beta_min}` to `{coarse_config.beta_max}` step `{
        coarse_config.beta_step
    }`
- grid points: `{len(coarse_results)}`
- runtime (seconds): `{coarse_runtime:.3f}`
- coarse max kendall_tau: `{coarse_results.iloc[0]["kendall_tau"]:.12f}`
- coarse maxima count: `{len(coarse_maxima)}`

## Refined Search

- refinement seeded from top `{config.refinement_seed_top_k}` coarse rows
- alpha range: `{refined_config.alpha_min}` to `{refined_config.alpha_max}` step `{
        refined_config.alpha_step
    }`
- beta range: `{refined_config.beta_min}` to `{refined_config.beta_max}` step `{
        refined_config.beta_step
    }`
- grid points: `{len(refined_results)}`
- runtime (seconds): `{refined_runtime:.3f}`
- refined max kendall_tau: `{refined_results.iloc[0]["kendall_tau"]:.12f}`
- refined maxima count: `{len(refined_maxima)}`

## Stability Analysis

- shortlisted refined maxima for stability: `{len(stability_candidates)}`
- bootstrap resamples: `{config.bootstrap_resamples}`
- bootstrap seed: `{config.bootstrap_seed}`
- runtime (seconds): `{stability_runtime:.3f}`
- note: {selection_note}

### Recommended Parameters

- alpha: `{recommended["alpha"]:.6f}`
- beta: `{recommended["beta"]:.6f}`
- full-sample kendall_tau: `{recommended["full_kendall_tau"]:.12f}`
- bootstrap mean kendall_tau: `{recommended["bootstrap_mean_kendall"]:.12f}`
- bootstrap win rate: `{recommended["bootstrap_win_rate"]:.12f}`
- leave-one-out mean kendall_tau: `{recommended["loo_mean_kendall"]:.12f}`
- full-sample spearman_rho: `{recommended["full_spearman_rho"]:.12f}`
- full-sample pearson_r: `{recommended["full_pearson_r"]:.12f}`

### Top {config.top_k} Stable Candidates

{
        format_table(
            stability_results.head(config.top_k),
            [
                "alpha",
                "beta",
                "full_kendall_tau",
                "bootstrap_mean_kendall",
                "bootstrap_win_rate",
                "loo_mean_kendall",
                "full_spearman_rho",
                "full_pearson_r",
            ],
        )
    }

## Artifacts

- coarse grid: `coarse_grid_results.csv`
- refined grid: `refined_grid_results.csv`
- refined maxima: `refined_maxima.csv`
- shortlisted stability candidates: `stability_candidates.csv`
- stability ranking: `stability_results.csv`
- coarse heatmap: `coarse/kendall_heatmap.png`
- refined heatmap: `refined/kendall_heatmap.png`
"""

    output_path = output_dir / "summary.md"
    output_path.write_text(summary)
    return output_path


def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(
        description="Robust optimization experiment for MEC(alpha, beta)."
    )
    parser.add_argument("--alpha-min", type=float, default=OptimizationConfig.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=OptimizationConfig.alpha_max)
    parser.add_argument(
        "--alpha-step", type=float, default=OptimizationConfig.alpha_step
    )
    parser.add_argument("--beta-min", type=float, default=OptimizationConfig.beta_min)
    parser.add_argument("--beta-max", type=float, default=OptimizationConfig.beta_max)
    parser.add_argument("--beta-step", type=float, default=OptimizationConfig.beta_step)
    parser.add_argument(
        "--refinement-seed-top-k",
        type=int,
        default=OptimizationConfig.refinement_seed_top_k,
    )
    parser.add_argument(
        "--refinement-alpha-step",
        type=float,
        default=OptimizationConfig.refinement_alpha_step,
    )
    parser.add_argument(
        "--refinement-beta-step",
        type=float,
        default=OptimizationConfig.refinement_beta_step,
    )
    parser.add_argument(
        "--stability-candidate-top-k",
        type=int,
        default=OptimizationConfig.stability_candidate_top_k,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=OptimizationConfig.bootstrap_resamples,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=OptimizationConfig.bootstrap_seed,
    )
    parser.add_argument("--top-k", type=int, default=OptimizationConfig.top_k)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OptimizationConfig.output_dir,
    )
    args = parser.parse_args()
    return OptimizationConfig(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        refinement_seed_top_k=args.refinement_seed_top_k,
        refinement_alpha_step=args.refinement_alpha_step,
        refinement_beta_step=args.refinement_beta_step,
        stability_candidate_top_k=args.stability_candidate_top_k,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )


def main(config: OptimizationConfig | None = None) -> None:
    config = config or parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coarse_config = build_search_config(
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        alpha_step=config.alpha_step,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        beta_step=config.beta_step,
        output_dir=output_dir / "coarse",
    )

    start = perf_counter()
    coarse_results = run_grid_search(coarse_config)
    coarse_runtime = perf_counter() - start
    (output_dir / "coarse").mkdir(parents=True, exist_ok=True)
    coarse_heatmap_path = save_heatmap(coarse_results, output_dir / "coarse")

    refined_config = build_refined_search_config(
        coarse_results=coarse_results,
        config=config,
        coarse_config=coarse_config,
        output_dir=output_dir / "refined",
    )

    start = perf_counter()
    refined_results = run_grid_search(refined_config)
    refined_runtime = perf_counter() - start
    (output_dir / "refined").mkdir(parents=True, exist_ok=True)
    refined_heatmap_path = save_heatmap(refined_results, output_dir / "refined")

    refined_maxima = find_kendall_maxima(refined_results)
    stability_candidates = refined_maxima.head(config.stability_candidate_top_k).copy()
    x_values, distributions, expert_scores = load_validation_problem()

    start = perf_counter()
    candidate_values = compute_values_grid(
        stability_candidates, x_values, distributions
    )
    stability_results = candidate_statistics(
        candidates=stability_candidates,
        candidate_values=candidate_values,
        expert_scores=expert_scores,
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
    )
    stability_runtime = perf_counter() - start

    coarse_results.to_csv(output_dir / "coarse_grid_results.csv", index=False)
    refined_results.to_csv(output_dir / "refined_grid_results.csv", index=False)
    refined_maxima.to_csv(output_dir / "refined_maxima.csv", index=False)
    stability_candidates.to_csv(output_dir / "stability_candidates.csv", index=False)
    stability_results.to_csv(output_dir / "stability_results.csv", index=False)
    summary_path = save_summary(
        config=config,
        coarse_config=coarse_config,
        refined_config=refined_config,
        coarse_results=coarse_results,
        refined_results=refined_results,
        stability_candidates=stability_candidates,
        stability_results=stability_results,
        coarse_runtime=coarse_runtime,
        refined_runtime=refined_runtime,
        stability_runtime=stability_runtime,
        output_dir=output_dir,
    )

    print(f"Saved coarse grid to {output_dir / 'coarse_grid_results.csv'}")
    print(f"Saved refined grid to {output_dir / 'refined_grid_results.csv'}")
    print(f"Saved refined maxima to {output_dir / 'refined_maxima.csv'}")
    print(f"Saved stability ranking to {output_dir / 'stability_results.csv'}")
    print(f"Saved coarse heatmap to {coarse_heatmap_path}")
    print(f"Saved refined heatmap to {refined_heatmap_path}")
    print(f"Saved summary to {summary_path}")
    print("\nRecommended parameters:")
    print(stability_results.iloc[0].to_dict())


if __name__ == "__main__":
    main()
