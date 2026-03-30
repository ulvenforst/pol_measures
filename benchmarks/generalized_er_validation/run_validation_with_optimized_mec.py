from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from benchmarks.expert_validation.data import ValidationData
from benchmarks.generalized_er_validation.run_validation import build_measures
from src.measures.metrics.proposed.mec import MECNormalized

OPTIMIZATION_RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "expert_validation"
    / "mec_parameter_optimization"
    / "stability_results.csv"
)
OUTPUT_PLOT = (
    Path(__file__).resolve().parent / "optimized_mec_vs_generalized_leaderboard.png"
)
OUTPUT_CSV = (
    Path(__file__).resolve().parent / "optimized_mec_vs_generalized_leaderboard.csv"
)
OUTPUT_REPORT = (
    Path(__file__).resolve().parent / "optimized_mec_vs_generalized_leaderboard.txt"
)


def load_best_parameters(results_path: Path) -> tuple[float, float]:
    results = pd.read_csv(results_path)
    if results.empty:
        raise ValueError(f"No optimization results found in {results_path}")
    best = results.iloc[0]
    return float(best["alpha"]), float(best["beta"])


def rename_generalized_measure(name: str) -> str:
    if name.startswith("ER(") and "," in name:
        return f"GER{name[2:]}"
    if (
        name.startswith("MEC(")
        and "," in name
        and any(token in name for token in ("d^", "exp(", "d+"))
    ):
        return f"GMEC{name[3:]}"
    return name


def build_measures_with_optimized_mec(
    best_alpha: float, best_beta: float
) -> dict[str, object]:
    measures = {}
    for name, measure in build_measures().items():
        measures[rename_generalized_measure(name)] = measure

    measures[f"MEC({best_alpha:.3f},{best_beta:.3f})"] = MECNormalized(
        alpha=best_alpha,
        beta=best_beta,
    )
    return measures


def compute_measure_values(
    measures: dict[str, object], x_values: np.ndarray, distributions: np.ndarray
) -> Dict[str, np.ndarray]:
    results: Dict[str, list[float]] = {name: [] for name in measures}

    for dist in distributions:
        for name, measure in measures.items():
            value = float(measure(x_values, dist, normalize_weights=True))
            results[name].append(value)

    return {name: np.array(vals, dtype=np.float64) for name, vals in results.items()}


def compute_correlations(
    measure_values: Dict[str, np.ndarray], expert_scores: np.ndarray
) -> pd.Series:
    correlations: dict[str, float] = {}
    for name, values in measure_values.items():
        correlations[name] = float(kendalltau(expert_scores, values).statistic)
    return pd.Series(correlations).sort_values(ascending=False)


def save_report(
    correlations: pd.Series,
    best_alpha: float,
    best_beta: float,
    output_path: Path,
) -> None:
    with output_path.open("w") as handle:
        handle.write("OPTIMIZED MEC VS GENERALIZED POLARIZATION LEADERBOARD\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(
            f"Optimized MEC parameters used: alpha={best_alpha:.6f}, beta={best_beta:.6f}\n\n"
        )
        for name, value in correlations.items():
            handle.write(f"{name}: {value:.12f}\n")


def plot_correlations(correlations: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(20, 7))
    ax = correlations.plot(kind="bar", color="#3279ad")
    plt.title("Correlacion de cada medida con la opinion de los expertos")
    plt.xlabel("Medidas")
    plt.ylabel("Correlacion de Kendall tau")

    for i, value in enumerate(correlations):
        ax.text(
            i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=6, rotation=90
        )

    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    best_alpha, best_beta = load_best_parameters(OPTIMIZATION_RESULTS_PATH)

    data = ValidationData()
    distributions = data.get_normalized_distributions()
    x_values = data.x_values
    expert_scores = data.expert_scores.astype(np.float64)

    measures = build_measures_with_optimized_mec(best_alpha, best_beta)
    measure_values = compute_measure_values(measures, x_values, distributions)
    correlations = compute_correlations(measure_values, expert_scores)

    correlations.to_csv(OUTPUT_CSV, header=["kendall_tau"])
    save_report(correlations, best_alpha, best_beta, OUTPUT_REPORT)
    plot_correlations(correlations, OUTPUT_PLOT)

    print(f"Optimized alpha={best_alpha:.6f}, beta={best_beta:.6f}")
    print(correlations.head(20))
    print(f"Saved plot to {OUTPUT_PLOT}")
    print(f"Saved csv to {OUTPUT_CSV}")
    print(f"Saved report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
