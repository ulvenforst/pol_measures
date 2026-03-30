from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.expert_validation.data import ValidationData
from src.measures.metrics.literature import EstebanRay, GeneralizedER
from src.measures.metrics.proposed.mec import MECNormalized

ALPHAS = [0.8, 1.0, 1.6]

ALIENATION_FUNCTIONS: List[Tuple[str, callable]] = [
    ("d^2", lambda d: d**2),
    ("d^3", lambda d: d**3),
    ("d+d^2", lambda d: d + d**2),
    ("d+2d^2", lambda d: d + 2 * d**2),
    ("exp(d)-1", lambda d: np.exp(d) - 1),
    ("exp(2d)-1", lambda d: np.exp(2 * d) - 1),
]


def build_measures() -> dict:
    measures = {
        "MEC(2,1.15)N": MECNormalized(),
        "ER(0.8)": EstebanRay(),
    }
    for alpha in ALPHAS:
        alpha_str = f"{alpha:.1f}" if alpha != 1.0 else "1"
        for fname, fn in ALIENATION_FUNCTIONS:
            key = f"ER({alpha_str},{fname})"
            measures[key] = GeneralizedER(alpha=alpha, alienation=fn)
    return measures


def compute_values(
    measures: dict, x_values: np.ndarray, distributions: np.ndarray
) -> Dict[str, np.ndarray]:
    results: Dict[str, list] = {name: [] for name in measures}
    for dist in distributions:
        for name, measure in measures.items():
            value = measure(x_values, dist, normalize_weights=True)
            results[name].append(np.trunc(value * 10000) / 10000)
    return {name: np.array(vals) for name, vals in results.items()}


def plot_combined(
    distributions: np.ndarray,
    expert_scores: np.ndarray,
    measure_values: Dict[str, np.ndarray],
) -> plt.Figure:
    """Single figure: table with mini distribution bar charts embedded in header cells."""
    measure_names = list(measure_values.keys())
    n_dists = len(distributions)
    n_measures = len(measure_names)
    n_rows_table = n_measures + 1  # +1 for Expert row

    # --- Build cell data and colours ---
    all_values = np.zeros((n_rows_table, n_dists))
    all_values[0] = expert_scores / 100
    for r, name in enumerate(measure_names):
        all_values[r + 1] = measure_values[name]

    row_mins = all_values.min(axis=1, keepdims=True)
    row_maxs = all_values.max(axis=1, keepdims=True)
    row_ranges = row_maxs - row_mins
    row_ranges[row_ranges == 0] = 1
    normalized = (all_values - row_mins) / row_ranges

    cmap_low = np.array([0.96, 0.97, 1.0])
    cmap_high = np.array([0.80, 0.20, 0.20])

    cell_text = []
    cell_colors = []
    row_labels = ["Expert"] + measure_names

    for r in range(n_rows_table):
        row_t = []
        row_c = []
        for c in range(n_dists):
            if r == 0:
                row_t.append(f"{expert_scores[c]:.1f}")
            else:
                row_t.append(f"{all_values[r][c]:.4f}")
            t = normalized[r][c]
            color = (1 - t) * cmap_low + t * cmap_high
            row_c.append(tuple(color))
        cell_text.append(row_t)
        cell_colors.append(row_c)

    # Header row: empty text (will be replaced by mini plots)
    header_text = [""] * n_dists
    header_colors = [("white",)] * n_dists

    # --- Figure ---
    header_row_height_factor = 4.5
    data_row_height = 1.35
    total_row_units = header_row_height_factor + n_rows_table * data_row_height
    fig_h = total_row_units * 0.28 + 1.2
    fig_w = 26
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes([0, 0, 1, 0.95])
    ax.axis("off")

    # Build full table: row 0 = header (for plots), rows 1.. = data
    full_cell_text = [header_text] + cell_text
    full_cell_colors = [[(1, 1, 1)] * n_dists] + cell_colors
    full_row_labels = [""] + row_labels

    table = ax.table(
        cellText=full_cell_text,
        rowLabels=full_row_labels,
        cellColours=full_cell_colors,
        rowColours=["#E8E8E8"] * (n_rows_table + 1),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # Scale: normal row height
    table.scale(1.0, data_row_height)

    label_col_w = 0.10

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)

        if col == -1:
            cell.set_text_props(fontweight="bold", fontsize=7, ha="right")
            cell.set_width(label_col_w)
            if row == 0:
                cell.set_height(cell.get_height() * header_row_height_factor)
                cell.set_linewidth(0)
                cell.set_facecolor("none")
                cell.set_edgecolor("none")
        else:
            data_col_w = (1.0 - label_col_w) / n_dists
            cell.set_width(data_col_w)

        # Header row (row 0): make tall for mini plots
        if row == 0 and col >= 0:
            cell.set_height(cell.get_height() * header_row_height_factor)
            cell.set_facecolor("white")
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.5)
            cell.get_text().set_text("")

        # Expert row (row 1): highlight
        if row == 1 and col >= 0:
            cell.set_text_props(fontweight="bold", fontsize=7.5)
            cell.set_facecolor("#F0E68C")

    # Draw to compute cell positions
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Embed mini bar charts inside header cells
    bar_color = "#4C72B0"
    categories = np.arange(1, 6)

    for c in range(n_dists):
        cell = table.get_celld()[(0, c)]
        bbox = cell.get_window_extent(renderer)
        bbox_fig = bbox.transformed(fig.transFigure.inverted())

        pad_x = bbox_fig.width * 0.08
        pad_y_bot = bbox_fig.height * 0.04
        pad_y_top = bbox_fig.height * 0.18

        ax_mini = fig.add_axes(
            [
                bbox_fig.x0 + pad_x,
                bbox_fig.y0 + pad_y_bot,
                bbox_fig.width - 2 * pad_x,
                bbox_fig.height - pad_y_bot - pad_y_top,
            ]
        )

        raw = distributions[c]
        freqs = raw / raw.sum() * 100
        ax_mini.bar(categories, freqs, color=bar_color, edgecolor="white", width=0.72)
        ax_mini.set_ylim(0, 80)
        ax_mini.set_xlim(0.3, 5.7)

        ax_mini.set_title(
            f"D{c + 1} | Exp: {expert_scores[c]:.1f}",
            fontsize=5.5,
            fontweight="bold",
            pad=1,
        )

        ax_mini.tick_params(
            axis="both",
            labelsize=0,
            length=0,
            bottom=False,
            left=False,
        )
        ax_mini.set_xticklabels([])
        ax_mini.set_yticklabels([])
        for spine in ax_mini.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color("#BBBBBB")

    fig.suptitle(
        "Polarization Values per Distribution: MEC(2,1.15)N, ER(0.8) & Generalized ER",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    return fig


def main():
    data = ValidationData()
    distributions = data.distributions
    normalized_dists = data.get_normalized_distributions()
    x_values = data.x_values
    expert_scores = data.expert_scores

    measures = build_measures()
    print(
        f"Computing {len(measures)} measures on {len(distributions)} distributions..."
    )
    measure_values = compute_values(measures, x_values, normalized_dists)

    fig = plot_combined(distributions, expert_scores, measure_values)
    fig.savefig(
        "benchmarks/generalized_er_validation/distributions_table.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved distributions_table.png")
    plt.show()


if __name__ == "__main__":
    main()
