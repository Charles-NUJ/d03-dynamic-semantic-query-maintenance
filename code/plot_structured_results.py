import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from figure_style import (
    FIGURE_SIZE,
    RC_PARAMS,
    SUBPLOT_ADJUST,
    format_axis,
    label_for,
    line_style_for,
)

METHOD_STYLE_KEYS = {
    "change_x_influence_ours": "signature_degree",
    "change_only_control": "input_change_id",
    "change_random_control": "input_change_random",
    "uniform_clean": "svc_uniform",
    "proxy_x_influence": "proxy_degree",
}

# Matplotlib fills a two-column legend down each column.  This permutation
# makes the row-wise reading order match METHOD_STYLE_KEYS: exact main,
# deterministic control, random control, SVC, and proxy.
LEGEND_DRAW_INDEX = [0, 2, 4, 1, 3]


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.results)
    budgets = [0.10, 0.25, 0.50, 0.75]

    plt.rcParams.update(RC_PARAMS)
    figure, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=False)
    metrics = [
        ("filter_error_fraction", "Filter output error (%)"),
        ("join_row_error_fraction", "Join output error (%)"),
    ]
    handles = []
    labels = []
    for method, style_key in METHOD_STYLE_KEYS.items():
        label = label_for(style_key)
        method_rows = {round(float(row["budget_fraction"]), 2): row for row in rows if row["method"] == method}
        x = [100 * budget for budget in budgets]
        for axis, (metric, ylabel) in zip(axes, metrics):
            y = [100 * float(method_rows[budget][metric]) for budget in budgets]
            plotted, = axis.plot(
                x,
                y,
                label=label,
                **line_style_for(style_key),
            )
            if axis is axes[0]:
                handles.append(plotted)
                labels.append(label)
            format_axis(
                axis,
                "Semantic-call budget (% of affected cells)",
                ylabel,
            )
            axis.set_xticks([10, 25, 50, 75])
            axis.set_xlim(8, 77)
    axes[0].set_ylim(-0.5, 16.0)
    axes[0].set_yticks([0, 5, 10, 15])
    axes[1].set_ylim(-1.0, 50.0)
    axes[1].set_yticks([0, 10, 20, 30, 40, 50])
    axes[0].set_title("(a) Filter output error", pad=2.0)
    axes[1].set_title("(b) Join output error", pad=2.0)
    legend_handles = [handles[index] for index in LEGEND_DRAW_INDEX]
    legend_labels = [labels[index] for index in LEGEND_DRAW_INDEX]
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.15,
        columnspacing=1.00,
        handletextpad=0.35,
        borderaxespad=0.05,
    )
    layout = dict(SUBPLOT_ADJUST)
    layout["top"] = 0.865
    figure.subplots_adjust(**layout)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf)
    figure.savefig(args.output_png, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
