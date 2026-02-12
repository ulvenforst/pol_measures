from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from benchmarks.expert_validation.data import ValidationData
from src.measures.metrics.literature import (
    EMDPol,
    EstebanRay,
    Experts,
    GeneralizedER,
    ShannonPol,
    VanDerEijkPol,
)
from src.measures.metrics.proposed import MEC, BiPol
from src.measures.metrics.proposed.mec import MECNormalized

ALPHAS = [0.8, 1.0, 1.6]

ALIENATION_FUNCTIONS = {
    "d^2": lambda d: d**2,
    "d^3": lambda d: d**3,
    "d+d^2": lambda d: d + d**2,
    "d+2d^2": lambda d: d + 2 * d**2,
    "exp(d)-1": lambda d: np.exp(d) - 1,
    "exp(2d)-1": lambda d: np.exp(2 * d) - 1,
}


def build_measures() -> dict:
    """Build the full dict of measures: existing ones + generalized ER variants."""
    measures = {
        "MEC(1,1)": MEC(alpha=1, beta=1),
        "MEC(2,1.15)": MECNormalized(),
        "MEC(2,1.2)": MEC(beta=1.2),
        "MEC(2,2)": MEC(alpha=2, beta=2),
        "EMD": EMDPol(),
        "ER(0.8)": EstebanRay(),
        "ER(1.6)": EstebanRay(alpha=1.6),
        "Experts": Experts(),
        "Shannon": ShannonPol(),
        "VanDerEijk": VanDerEijkPol(),
        "BiPol": BiPol(),
    }

    for alpha in ALPHAS:
        alpha_str = f"{alpha:.1f}" if alpha != 1.0 else "1"
        for fname, fn in ALIENATION_FUNCTIONS.items():
            key = f"ER({alpha_str},{fname})"
            measures[key] = GeneralizedER(alpha=alpha, alienation=fn)

    return measures


def compute_measure_values(
    measures: dict, x_values: np.ndarray, distributions: np.ndarray
) -> Dict[str, np.ndarray]:
    """Process all distributions through all measures, return {name: array_of_values}."""
    results: Dict[str, list] = {name: [] for name in measures}

    for dist in distributions:
        for name, measure in measures.items():
            value = measure(x_values, dist)
            results[name].append(np.trunc(value * 10000) / 10000)

    return {name: np.array(vals) for name, vals in results.items()}


def compute_correlations(
    measure_values: Dict[str, np.ndarray], expert_scores: np.ndarray
) -> pd.Series:
    """Compute Kendall tau between each measure and expert scores."""
    correlations = {}
    for name, values in measure_values.items():
        tau, _ = kendalltau(expert_scores, values)
        correlations[name] = np.trunc(tau * 10000) / 10000
    return pd.Series(correlations).sort_values(ascending=False)


def save_results(
    measure_values: Dict[str, np.ndarray],
    expert_scores: np.ndarray,
    distributions: np.ndarray,
    filename: str = "benchmarks/generalized_er_validation/generalized_er_valores.txt",
) -> None:
    """Save polarization values and correlations to a text file."""
    with open(filename, "w") as f:
        f.write("GENERALIZED ER VALIDATION - POLARIZATION VALUES\n")
        f.write("=" * 60 + "\n\n")

        f.write("Distributions:\n")
        for i, dist in enumerate(distributions, 1):
            f.write(f"Distribution {i}: {dist}\n")
        f.write("\n" + "=" * 60 + "\n\n")

        f.write("Expert scores:\n")
        for i, val in enumerate(expert_scores, 1):
            f.write(f"Distribution {i}: {val:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")

        for name, values in measure_values.items():
            f.write(f"\n{name}:\n")
            f.write("-" * 40 + "\n")
            for i, val in enumerate(values, 1):
                f.write(f"Distribution {i}: {val:.4f}\n")
            tau, _ = kendalltau(expert_scores, values)
            f.write(f"\nKendall tau with experts: {tau:.4f}\n")
            f.write("=" * 60 + "\n")


def plot_correlations(correlations: pd.Series) -> None:
    """Plot a bar chart of Kendall tau correlations (without RealMeanExp)."""
    plot_data = correlations.drop("RealMeanExp", errors="ignore")

    plt.figure(figsize=(20, 7))
    ax = plot_data.plot(kind="bar")
    plt.title("Correlation with expert judgments (60 experts) - Generalized ER")
    plt.xlabel("Measure")
    plt.ylabel("Kendall rank correlation coefficient")

    for i, v in enumerate(plot_data):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=6, rotation=90)

    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig("benchmarks/generalized_er_validation/Figure_1.png", dpi=150)
    plt.show()


def main():
    data = ValidationData()
    distributions = data.get_normalized_distributions()
    x_values = data.x_values
    expert_scores = np.trunc(data.expert_scores * 10000) / 10000

    measures = build_measures()

    print(f"Running {len(measures)} measures on {len(distributions)} distributions...")
    measure_values = compute_measure_values(measures, x_values, distributions)
    measure_values["RealMeanExp"] = expert_scores

    save_results(measure_values, expert_scores, distributions)

    correlations = compute_correlations(measure_values, expert_scores)
    print("\nKendall tau correlations with expert scores:")
    print(correlations)

    plot_correlations(correlations)


if __name__ == "__main__":
    main()
