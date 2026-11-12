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
    "change_degree": "signature_degree",
    "random_changed": "input_change_random",
    "random_all": "svc_uniform",
    "degree_all": "link_degree",
}

# Matplotlib fills a two-column legend down each column.  This permutation
# preserves the row-wise factor order used in the experiment description.
LEGEND_DRAW_INDEX = [0, 2, 1, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    plt.rcParams.update(RC_PARAMS)
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=False)
    for method, style_key in METHOD_STYLE_KEYS.items():
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["budget"]),
        )
        x = [100.0 * int(row["budget"]) / 320.0 for row in selected]
        error = [100.0 * float(row["join_row_error_fraction"]) for row in selected]
        coverage = [100.0 * float(row["changed_degree_coverage"]) for row in selected]
        label = label_for(style_key)
        axes[0].plot(x, error, label=label, **line_style_for(style_key))
        axes[1].plot(x, coverage, label=label, **line_style_for(style_key))

    threshold = 100.0 * 145.0 / 320.0
    for axis in axes:
        axis.axvline(threshold, color="0.65", linestyle=(0, (2, 2)), linewidth=0.9)
        axis.set_xlim(-1.0, 51.5)
        axis.set_xticks([0, 10, 20, 30, 40, 50])

    axes[0].set_ylim(-1, 55)
    format_axis(
        axes[0],
        "Semantic-call budget (% of affected cells)",
        "Join output error (%)",
    )
    axes[0].set_yticks([0, 10, 20, 30, 40, 50])
    axes[0].set_title("(a) Join output error", pad=2.0)
    axes[1].set_ylim(-2, 104)
    format_axis(
        axes[1],
        "Semantic-call budget (% of affected cells)",
        "Changed-degree coverage (%)",
    )
    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_title("(b) Changed-degree coverage", pad=2.0)
    axes[1].text(
        threshold - 0.9,
        3,
        r"$B=|C|$",
        fontsize=6.8,
        rotation=90,
        va="bottom",
        ha="right",
        color="0.25",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
    )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    handles = [handles[index] for index in LEGEND_DRAW_INDEX]
    legend_labels = [legend_labels[index] for index in LEGEND_DRAW_INDEX]
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=1.05,
        handlelength=2.35,
        handletextpad=0.40,
        borderaxespad=0.05,
    )
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf)
    fig.savefig(args.output_png, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
