from __future__ import annotations

import sys
from dataclasses import dataclass
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

if __package__ is None or __package__ == "":
    from benchmarks.expert_validation.data import ValidationData
else:
    from .data import ValidationData

from src.measures.metrics.literature import (
    EMDPol,
    EstebanRay,
    Experts,
    ShannonPol,
    VanDerEijkPol,
)
from src.measures.metrics.proposed import BiPol
from src.measures.metrics.proposed.mec import MEC, MECNormalized


@dataclass(frozen=True)
class OptimizedValidationConfig:
    optimization_results_path: Path = (
        Path(__file__).resolve().parent
        / "mec_parameter_optimization"
        / "stability_results.csv"
    )
    output_dir: Path = (
        Path(__file__).resolve().parent / "optimized_validation_against_experts"
    )


def load_best_parameters(results_path: Path) -> tuple[float, float]:
    results = pd.read_csv(results_path)
    if results.empty:
        raise ValueError(f"No optimization results found in {results_path}")
    best = results.iloc[0]
    return float(best["alpha"]), float(best["beta"])


def stable_order(values: np.ndarray) -> np.ndarray:
    return np.argsort(values, kind="stable") + 1


def order_to_rank(order: np.ndarray) -> np.ndarray:
    rank = np.empty_like(order)
    for position, distribution_id in enumerate(order, start=1):
        rank[distribution_id - 1] = position
    return rank


def kendall_against_expert_order(
    expert_scores: np.ndarray, values: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    expert_order = stable_order(expert_scores)
    value_order = stable_order(values)
    expert_rank = order_to_rank(expert_order)
    value_rank = order_to_rank(value_order)
    tau = float(kendalltau(expert_rank, value_rank).statistic)
    return tau, expert_order, value_order


def build_measures(best_alpha: float, best_beta: float) -> dict[str, object]:
    return {
        "MEC(opt)": MECNormalized(alpha=best_alpha, beta=best_beta),
        "MEC(2,1.15)": MECNormalized(),
        "MEC(1,1)": MEC(alpha=1, beta=1),
        "MEC(2,1.2)": MEC(alpha=2, beta=1.2),
        "MEC(2,2)": MEC(alpha=2, beta=2),
        "ER(0.8)": EstebanRay(),
        "ER(1.6)": EstebanRay(alpha=1.6),
        "Experts": Experts(),
        "BiPol": BiPol(),
        "Shannon": ShannonPol(),
        "EMD": EMDPol(),
        "VanDerEijk": VanDerEijkPol(),
    }


def compute_measure_values(
    measures: dict[str, object], x_values: np.ndarray, distributions: np.ndarray
) -> Dict[str, np.ndarray]:
    results: Dict[str, list[float]] = {name: [] for name in measures}
    for dist in distributions:
        for name, measure in measures.items():
            value = float(measure(x_values, dist, normalize_weights=True))
            results[name].append(value)
    return {
        name: np.array(values, dtype=np.float64) for name, values in results.items()
    }


def compute_correlations(
    measure_values: Dict[str, np.ndarray], expert_scores: np.ndarray
) -> tuple[pd.Series, dict[str, dict[str, np.ndarray | float]]]:
    correlations: dict[str, float] = {"RealMeanExp": 1.0}
    ranking_details: dict[str, dict[str, np.ndarray | float]] = {}

    expert_tau, expert_order, expert_value_order = kendall_against_expert_order(
        expert_scores, expert_scores
    )
    ranking_details["RealMeanExp"] = {
        "kendall_tau": expert_tau,
        "expert_order": expert_order,
        "value_order": expert_value_order,
        "values": expert_scores,
    }

    for name, values in measure_values.items():
        tau, expert_order, value_order = kendall_against_expert_order(
            expert_scores, values
        )
        correlations[name] = tau
        ranking_details[name] = {
            "kendall_tau": tau,
            "expert_order": expert_order,
            "value_order": value_order,
            "values": values,
        }

    series = pd.Series(correlations).sort_values(ascending=False)
    return series, ranking_details


def save_detailed_report(
    ranking_details: dict[str, dict[str, np.ndarray | float]],
    best_alpha: float,
    best_beta: float,
    output_path: Path,
) -> None:
    with output_path.open("w") as handle:
        handle.write("OPTIMIZED VALIDATION AGAINST THE EXPERT ORDER\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(
            f"Optimized MEC parameters: alpha={best_alpha:.6f}, beta={best_beta:.6f}\n\n"
        )

        for measure_name, details in ranking_details.items():
            values = np.asarray(details["values"])
            expert_order = np.asarray(details["expert_order"])
            value_order = np.asarray(details["value_order"])
            tau = float(details["kendall_tau"])

            handle.write(f"{measure_name}:\n")
            handle.write("-" * 40 + "\n")
            handle.write("Values in original order:\n")
            for idx, value in enumerate(values, start=1):
                handle.write(f"Distribution {idx}: {value:.6f}\n")
            handle.write("\nExpert order:\n")
            handle.write(f"{expert_order.tolist()}\n")
            handle.write("Produced order:\n")
            handle.write(f"{value_order.tolist()}\n")
            handle.write(f"Kendall tau with expert order: {tau:.12f}\n")
            handle.write("\n" + "=" * 60 + "\n\n")


def plot_correlations(
    correlations: pd.Series,
    best_alpha: float,
    best_beta: float,
    output_path: Path,
) -> None:
    labels = correlations.index.tolist()
    labels = [
        f"MEC({best_alpha:.3f},{best_beta:.3f})" if label == "MEC(opt)" else label
        for label in labels
    ]

    plt.figure(figsize=(13, 6))
    bars = plt.bar(labels, correlations.values, color="#3279ad")
    plt.title("Correlación de cada medida con el orden de los expertos")
    plt.xlabel("Medidas")
    plt.ylabel("Correlación de Kendall tau")

    for bar, value in zip(bars, correlations.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main(config: OptimizedValidationConfig | None = None) -> None:
    config = config or OptimizedValidationConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    best_alpha, best_beta = load_best_parameters(config.optimization_results_path)

    data = ValidationData()
    distributions = data.get_normalized_distributions()
    x_values = data.x_values
    expert_scores = data.expert_scores.astype(np.float64)

    measures = build_measures(best_alpha, best_beta)
    measure_values = compute_measure_values(measures, x_values, distributions)
    correlations, ranking_details = compute_correlations(measure_values, expert_scores)

    plot_correlations(
        correlations=correlations,
        best_alpha=best_alpha,
        best_beta=best_beta,
        output_path=config.output_dir / "optimized_validation_correlations.png",
    )
    save_detailed_report(
        ranking_details=ranking_details,
        best_alpha=best_alpha,
        best_beta=best_beta,
        output_path=config.output_dir / "optimized_validation_details.txt",
    )
    correlations.to_csv(config.output_dir / "optimized_validation_correlations.csv")

    print(f"Optimized alpha={best_alpha:.6f}, beta={best_beta:.6f}")
    print(correlations)
    print(
        f"Saved figure to {config.output_dir / 'optimized_validation_correlations.png'}"
    )
    print(f"Saved report to {config.output_dir / 'optimized_validation_details.txt'}")


if __name__ == "__main__":
    main()
